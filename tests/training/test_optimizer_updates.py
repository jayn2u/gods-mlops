from __future__ import annotations

import io
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

import torch

from gods_mlops.training.runner_support import step_optimizer


class _TinyLossModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))
        self.register_buffer("forward_buffer", torch.tensor(0))
        self.forward_calls = 0

    def forward(self, value: torch.Tensor) -> SimpleNamespace:
        self.forward_calls += 1
        self.forward_buffer.add_(1)
        return SimpleNamespace(loss=(self.weight * value).square().sum())


class _CapturedClipLogitsModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(
            torch.tensor(
                [[29.984375, 5.48828125], [13.34375, 30.03125]],
                dtype=torch.float32,
            )
        )

    def forward(self) -> SimpleNamespace:
        return SimpleNamespace(
            loss=None,
            logits_per_image=self.logits.to(dtype=torch.float16),
        )


class _ZeroGradientModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))

    def forward(self) -> SimpleNamespace:
        return SimpleNamespace(loss=self.weight.sum() * 0 + 1)


class _CountingSGD(torch.optim.SGD):
    def __init__(self, parameters, *, lr: float) -> None:
        super().__init__(parameters, lr=lr)
        self.step_calls = 0

    def step(self, *args, **kwargs):
        self.step_calls += 1
        return super().step(*args, **kwargs)


class _FakeGradScaler:
    """CPU stand-in that matches GradScaler's skip-and-backoff behavior."""

    def __init__(self, *, skip_attempts: int, scale: float = 8.0) -> None:
        self.skip_attempts = skip_attempts
        self.scale_value = scale
        self.skip_this_attempt = False

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss * self.scale_value

    def unscale_(self, optimizer) -> None:
        gradients = [
            parameter.grad
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.grad is not None
        ]
        for gradient in gradients:
            gradient.div_(self.scale_value)
        self.skip_this_attempt = self.skip_attempts > 0
        if self.skip_this_attempt:
            self.skip_attempts -= 1
            gradients[0].fill_(float("inf"))

    def step(self, optimizer) -> None:
        if not self.skip_this_attempt:
            optimizer.step()
        return None

    def update(self) -> None:
        if self.skip_this_attempt:
            self.scale_value *= 0.5
        self.skip_this_attempt = False

    def get_scale(self) -> float:
        return self.scale_value

    def state_dict(self) -> dict[str, float | int]:
        return {"scale": self.scale_value, "skip_attempts": self.skip_attempts}

    def load_state_dict(self, state: dict[str, float | int]) -> None:
        self.scale_value = float(state["scale"])
        self.skip_attempts = int(state["skip_attempts"])


class OptimizerUpdateTests(unittest.TestCase):
    def test_clip_fp64_symmetric_loss_preserves_the_captured_positive_signal(self) -> None:
        from gods_mlops.training.clip import _symmetric_clip_loss_from_output

        logits = torch.tensor(
            [[29.984375, 5.48828125], [13.34375, 30.03125]],
            dtype=torch.float16,
            requires_grad=True,
        )
        loss = _symmetric_clip_loss_from_output(SimpleNamespace(logits_per_image=logits))

        self.assertEqual(loss.dtype, torch.float64)
        self.assertAlmostEqual(loss.item(), 2.898325678739593e-8, delta=1e-15)
        loss.backward()
        self.assertIsNotNone(logits.grad)

    def test_clip_fp64_loss_matches_the_stock_symmetric_objective_on_unsaturated_logits(self) -> None:
        from torch.nn import functional as F

        from gods_mlops.training.clip import _symmetric_clip_loss_from_output

        logits = torch.tensor(
            [[2.0, -1.0, 0.5], [-0.5, 1.25, 0.0], [0.2, -0.7, 1.5]],
            dtype=torch.float32,
            requires_grad=True,
        )
        targets = torch.arange(3)
        expected = (
            F.cross_entropy(logits.to(torch.float64), targets)
            + F.cross_entropy(logits.transpose(0, 1).to(torch.float64), targets)
        ) / 2

        actual = _symmetric_clip_loss_from_output(
            SimpleNamespace(logits_per_image=logits)
        )

        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(actual.requires_grad)

    def test_step_optimizer_selects_attached_loss_after_autocast_and_guards_real_gradients(self) -> None:
        from gods_mlops.training import runner_support
        from gods_mlops.training.clip import _symmetric_clip_loss_from_output

        model = _CapturedClipLogitsModel()
        optimizer = _CountingSGD(model.parameters(), lr=0.1)
        scaler = _FakeGradScaler(skip_attempts=0, scale=65536.0)
        stats = runner_support.OptimizerStepStats()
        autocast_state = {"active": False}

        class AutocastContext:
            def __enter__(self):
                autocast_state["active"] = True

            def __exit__(self, *_args):
                autocast_state["active"] = False

        def loss_from_output(output):
            self.assertFalse(autocast_state["active"])
            return _symmetric_clip_loss_from_output(output)

        with (
            mock.patch.object(torch, "autocast", side_effect=lambda **_kwargs: AutocastContext()),
            mock.patch.object(torch.cuda, "synchronize", return_value=None),
        ):
            loss = runner_support.step_optimizer(
                model,
                optimizer,
                {},
                scaler=scaler,
                loss_from_output=loss_from_output,
                require_finite_nonzero_gradients=True,
                step_stats=stats,
            )

        self.assertGreater(loss, 0)
        self.assertEqual(optimizer.step_calls, 1)
        self.assertEqual(stats.finite_nonzero_gradient_updates, 1)
        self.assertGreater(stats.max_nonzero_trainable_parameters, 0)

    def test_clip_gradient_guard_allows_overflow_retry_but_counts_only_successful_gradients(self) -> None:
        from gods_mlops.training import runner_support
        from gods_mlops.training.clip import _symmetric_clip_loss_from_output

        model = _CapturedClipLogitsModel()
        optimizer = _CountingSGD(model.parameters(), lr=0.1)
        scaler = _FakeGradScaler(skip_attempts=1, scale=65536.0)
        stats = runner_support.OptimizerStepStats()

        with (
            mock.patch.object(torch, "autocast", side_effect=lambda **_kwargs: nullcontext()),
            mock.patch.object(torch.cuda, "synchronize", return_value=None),
        ):
            runner_support.step_optimizer(
                model,
                optimizer,
                {},
                scaler=scaler,
                loss_from_output=_symmetric_clip_loss_from_output,
                require_finite_nonzero_gradients=True,
                step_stats=stats,
            )

        self.assertEqual(optimizer.step_calls, 1)
        self.assertEqual(stats.attempts, 2)
        self.assertEqual(stats.amp_overflow_skips, 1)
        self.assertEqual(stats.finite_nonzero_gradient_updates, 1)
        self.assertGreater(stats.max_nonzero_trainable_parameters, 0)

    def test_clip_gradient_guard_rejects_positive_loss_with_only_zero_parameter_gradients(self) -> None:
        from gods_mlops.training import runner_support

        model = _ZeroGradientModel()
        optimizer = _CountingSGD(model.parameters(), lr=0.1)
        scaler = _FakeGradScaler(skip_attempts=0)
        stats = runner_support.OptimizerStepStats()

        with (
            mock.patch.object(torch, "autocast", side_effect=lambda **_kwargs: nullcontext()),
            mock.patch.object(torch.cuda, "synchronize", return_value=None),
        ):
            with self.assertRaisesRegex(ValueError, "finite nonzero trainable parameter gradients"):
                runner_support.step_optimizer(
                    model,
                    optimizer,
                    {},
                    scaler=scaler,
                    require_finite_nonzero_gradients=True,
                    step_stats=stats,
                )

        self.assertEqual(optimizer.step_calls, 0)
        self.assertEqual(stats.attempts, 1)
        self.assertEqual(stats.amp_overflow_skips, 0)

    def test_step_optimizer_keeps_missing_zero_and_nonfinite_selected_loss_guards(self) -> None:
        from gods_mlops.training import runner_support

        class CountingScaler(_FakeGradScaler):
            def __init__(self) -> None:
                super().__init__(skip_attempts=0)
                self.scale_calls = 0

            def scale(self, loss: torch.Tensor) -> torch.Tensor:
                self.scale_calls += 1
                return super().scale(loss)

        for selected_loss in (None, torch.tensor(0.0), torch.tensor(float("nan"))):
            with self.subTest(selected_loss=selected_loss):
                model = _TinyLossModel()
                optimizer = _CountingSGD(model.parameters(), lr=0.1)
                scaler = CountingScaler()
                stats = runner_support.OptimizerStepStats()
                with (
                    mock.patch.object(torch, "autocast", side_effect=lambda **_kwargs: nullcontext()),
                    mock.patch.object(torch.cuda, "synchronize", return_value=None),
                ):
                    with self.assertRaises(ValueError):
                        runner_support.step_optimizer(
                            model,
                            optimizer,
                            {"value": torch.tensor([1.0])},
                            scaler=scaler,
                            loss_from_output=lambda _output: selected_loss,
                            step_stats=stats,
                        )
                self.assertEqual(scaler.scale_calls, 0)
                self.assertEqual(optimizer.step_calls, 0)
                self.assertEqual(stats.attempts, 1)
                self.assertEqual(stats.amp_overflow_skips, 0)

    def test_step_optimizer_retries_a_scaler_skip_until_adamw_runs(self) -> None:
        from gods_mlops.training import runner_support

        model = _TinyLossModel()
        optimizer = _CountingSGD(model.parameters(), lr=0.1)
        scaler = _FakeGradScaler(skip_attempts=1)
        stats = runner_support.OptimizerStepStats()
        original_weight = model.weight.detach().clone()

        with (
            mock.patch.object(torch, "autocast", side_effect=lambda **_kwargs: nullcontext()),
            mock.patch.object(torch.cuda, "synchronize", return_value=None),
        ):
            step_optimizer(
                model,
                optimizer,
                {"value": torch.tensor([1.0])},
                scaler=scaler,
                step_stats=stats,
            )

        self.assertEqual(optimizer.step_calls, 1)
        self.assertEqual(model.forward_calls, 2)
        self.assertEqual(int(model.forward_buffer.item()), 1)
        self.assertEqual(stats.attempts, 2)
        self.assertEqual(stats.amp_overflow_skips, 1)
        self.assertFalse(torch.equal(original_weight, model.weight.detach()))

    def test_perpetual_scaler_skips_exhaust_the_bound_without_an_optimizer_step(self) -> None:
        from gods_mlops.training import runner_support

        model = _TinyLossModel()
        optimizer = _CountingSGD(model.parameters(), lr=0.1)
        scaler = _FakeGradScaler(skip_attempts=100)
        stats = runner_support.OptimizerStepStats()
        original_weight = model.weight.detach().clone()

        with (
            mock.patch.object(torch, "autocast", side_effect=lambda **_kwargs: nullcontext()),
            mock.patch.object(torch.cuda, "synchronize", return_value=None),
        ):
            with self.assertRaisesRegex(
                ValueError,
                f"{runner_support.MAX_AMP_STEP_ATTEMPTS} attempts",
            ):
                step_optimizer(
                    model,
                    optimizer,
                    {"value": torch.tensor([1.0])},
                    scaler=scaler,
                    step_stats=stats,
                )

        self.assertEqual(runner_support.MAX_AMP_STEP_ATTEMPTS, 32)
        self.assertEqual(stats.attempts, 32)
        self.assertEqual(stats.amp_overflow_skips, 32)
        self.assertEqual(model.forward_calls, 32)
        self.assertEqual(int(model.forward_buffer.item()), 0)
        self.assertEqual(optimizer.step_calls, 0)
        self.assertTrue(torch.equal(original_weight, model.weight.detach()))

    def test_retry_guard_yields_before_running_the_next_forward(self) -> None:
        from gods_mlops.training import runner_support
        from gods_mlops.training.claims import WorkerYieldRequested

        model = _TinyLossModel()
        optimizer = _CountingSGD(model.parameters(), lr=0.1)
        scaler = _FakeGradScaler(skip_attempts=1)
        stats = runner_support.OptimizerStepStats()
        guard_calls = 0

        def before_attempt() -> None:
            nonlocal guard_calls
            guard_calls += 1
            if guard_calls == 2:
                raise WorkerYieldRequested("test yield at AMP retry boundary")

        with (
            mock.patch.object(torch, "autocast", side_effect=lambda **_kwargs: nullcontext()),
            mock.patch.object(torch.cuda, "synchronize", return_value=None),
        ):
            with self.assertRaises(WorkerYieldRequested):
                step_optimizer(
                    model,
                    optimizer,
                    {"value": torch.tensor([1.0])},
                    scaler=scaler,
                    step_stats=stats,
                    before_attempt=before_attempt,
                )

        self.assertEqual(guard_calls, 2)
        self.assertEqual(model.forward_calls, 1)
        self.assertEqual(int(model.forward_buffer.item()), 0)
        self.assertEqual(stats.attempts, 1)
        self.assertEqual(stats.amp_overflow_skips, 1)
        self.assertEqual(optimizer.step_calls, 0)

    def test_checkpoint_round_trip_preserves_amp_retry_counters(self) -> None:
        from gods_mlops.jobs.checkpoints import CheckpointIdentity
        from gods_mlops.training import runner_support

        model = _TinyLossModel()
        optimizer = _CountingSGD(model.parameters(), lr=0.1)
        scaler = _FakeGradScaler(skip_attempts=1)
        stats = runner_support.OptimizerStepStats()
        identity = CheckpointIdentity(
            job_id="a0320b59-663c-4cdc-b893-086bb970ea60",
            input_kind="probe_input",
            input_id="task8-detr-probe-v1",
            input_sha256="1" * 64,
            phase="probe",
            model_kind="detr",
            config_version="task8-detr-config-v1",
            config_sha256="2" * 64,
            dataset_version=None,
        )

        with (
            mock.patch.object(torch, "autocast", side_effect=lambda **_kwargs: nullcontext()),
            mock.patch.object(torch.cuda, "synchronize", return_value=None),
        ):
            step_optimizer(
                model,
                optimizer,
                {"value": torch.tensor([1.0])},
                scaler=scaler,
                step_stats=stats,
            )

        payload = runner_support.serialize_checkpoint(
            model=model,
            optimizer_instance=optimizer,
            optimizer_steps=1,
            identity=identity,
            model_revision="revision-1",
            precision="float16-autocast",
            scaler=scaler,
            step_stats=stats,
        )
        resumed_model = _TinyLossModel()
        resumed_optimizer = _CountingSGD(resumed_model.parameters(), lr=0.1)
        resumed_scaler = _FakeGradScaler(skip_attempts=0)
        resumed_stats = runner_support.OptimizerStepStats()

        resumed_step = runner_support.restore_checkpoint(
            payload,
            model=resumed_model,
            optimizer_instance=resumed_optimizer,
            expected_identity=identity,
            expected_revision="revision-1",
            device=torch.device("cpu"),
            scaler=resumed_scaler,
            step_stats=resumed_stats,
        )

        self.assertEqual(resumed_step, 1)
        self.assertEqual(resumed_stats.attempts, 2)
        self.assertEqual(resumed_stats.amp_overflow_skips, 1)

    def test_clip_checkpoint_records_policy_and_rejects_old_or_mismatched_policy_before_loading(self) -> None:
        from gods_mlops.jobs.checkpoints import CheckpointIdentity
        from gods_mlops.training import runner_support

        policy = {
            "forward_precision": "float16-autocast",
            "training_loss_policy_version": "task9-clip-symmetric-ce-fp64-v1",
            "training_loss_objective": "symmetric_identity_cross_entropy",
            "training_loss_reduction_precision": "float64",
        }
        model = _CapturedClipLogitsModel()
        optimizer = _CountingSGD(model.parameters(), lr=0.1)
        scaler = _FakeGradScaler(skip_attempts=0)
        stats = runner_support.OptimizerStepStats(
            attempts=1,
            finite_nonzero_gradient_updates=1,
            max_nonzero_trainable_parameters=1,
        )
        old_identity = CheckpointIdentity(
            job_id="e4d82c26-8f0a-4f45-8b88-5fe84302d948",
            input_kind="probe_input",
            input_id="task8-clip-synthetic-probe-v1",
            input_sha256="b9f04dcf067c0dc4d53fa4b39bd88a991b9f956ca91590ba6d95a2293416731e",
            phase="probe",
            model_kind="clip",
            config_version="task8-clip-224-microbatch2-explicit-negative-probe-v1",
            config_sha256="f9dab1521d78a8297c2c0ae29bfa1de7b5a60d3f114435d96ca58a97bb2a0a6f",
            dataset_version=None,
        )
        new_identity = CheckpointIdentity(
            job_id=old_identity.job_id,
            input_kind=old_identity.input_kind,
            input_id=old_identity.input_id,
            input_sha256=old_identity.input_sha256,
            phase=old_identity.phase,
            model_kind=old_identity.model_kind,
            config_version="task9-clip-224-microbatch2-symmetric-ce-fp64-probe-v1",
            config_sha256="9" * 64,
            dataset_version=None,
        )
        payload = runner_support.serialize_checkpoint(
            model=model,
            optimizer_instance=optimizer,
            optimizer_steps=1,
            identity=new_identity,
            model_revision="57c216476eefef5ab752ec549e440a49ae4ae5f3",
            precision="float16-autocast",
            scaler=scaler,
            step_stats=stats,
            execution_policy=policy,
        )
        checkpoint = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
        self.assertEqual(checkpoint["execution_policy"], policy)
        self.assertEqual(checkpoint["finite_nonzero_gradient_updates"], 1)
        self.assertEqual(checkpoint["max_nonzero_trainable_parameters"], 1)

        resumed_model = _CapturedClipLogitsModel()
        resumed_optimizer = _CountingSGD(resumed_model.parameters(), lr=0.1)
        resumed_scaler = _FakeGradScaler(skip_attempts=0)
        resumed_stats = runner_support.OptimizerStepStats()
        self.assertEqual(
            runner_support.restore_checkpoint(
                payload,
                model=resumed_model,
                optimizer_instance=resumed_optimizer,
                expected_identity=new_identity,
                expected_revision="57c216476eefef5ab752ec549e440a49ae4ae5f3",
                device=torch.device("cpu"),
                scaler=resumed_scaler,
                step_stats=resumed_stats,
                expected_execution_policy=policy,
            ),
            1,
        )
        self.assertEqual(resumed_stats.finite_nonzero_gradient_updates, 1)
        self.assertEqual(resumed_stats.max_nonzero_trainable_parameters, 1)

        wrong_policy = {**policy, "training_loss_reduction_precision": "float32"}
        with (
            mock.patch.object(resumed_model, "load_state_dict", wraps=resumed_model.load_state_dict) as load_model,
            self.assertRaisesRegex(ValueError, "execution policy"),
        ):
            runner_support.restore_checkpoint(
                payload,
                model=resumed_model,
                optimizer_instance=resumed_optimizer,
                expected_identity=new_identity,
                expected_revision="57c216476eefef5ab752ec549e440a49ae4ae5f3",
                device=torch.device("cpu"),
                scaler=resumed_scaler,
                expected_execution_policy=wrong_policy,
            )
        load_model.assert_not_called()

        old_payload = runner_support.serialize_checkpoint(
            model=model,
            optimizer_instance=optimizer,
            optimizer_steps=1,
            identity=old_identity,
            model_revision="57c216476eefef5ab752ec549e440a49ae4ae5f3",
            precision="float16-autocast",
            scaler=scaler,
            step_stats=runner_support.OptimizerStepStats(attempts=1),
        )
        with self.assertRaisesRegex(ValueError, "identity changed"):
            runner_support.restore_checkpoint(
                old_payload,
                model=resumed_model,
                optimizer_instance=resumed_optimizer,
                expected_identity=new_identity,
                expected_revision="57c216476eefef5ab752ec549e440a49ae4ae5f3",
                device=torch.device("cpu"),
                scaler=resumed_scaler,
                expected_execution_policy=policy,
            )


if __name__ == "__main__":
    unittest.main()
