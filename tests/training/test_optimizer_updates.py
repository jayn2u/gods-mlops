from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()
