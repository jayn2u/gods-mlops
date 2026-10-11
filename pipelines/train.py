"""Published-dataset training pipeline; Task 7 owns all GPU waiting/admission."""

import argparse
import os
from pathlib import Path

from kfp import compiler, dsl

from .components import configure_controller_task, controller_component, validate_training_image


def build_pipeline(training_image: str):
    train_component, _prepare_component = controller_component(training_image)

    @dsl.pipeline(name="gods-mlops-published-training")
    def train_pipeline(dataset_version: str, detr_config_version: str, clip_config_version: str):
        detr = train_component(
            dataset_version=dataset_version,
            model_kind="detr",
            config_version=detr_config_version,
        )
        configure_controller_task(detr, training_image=training_image)
        detr.set_display_name("RT-DETR person training")

        clip = train_component(
            dataset_version=dataset_version,
            model_kind="clip",
            config_version=clip_config_version,
        )
        configure_controller_task(clip, training_image=training_image)
        clip.set_display_name("CLIP crop-text training")
        clip.after(detr)

    return train_pipeline


def compile_pipeline(destination: str | Path, *, training_image: str | None = None) -> Path:
    image = validate_training_image(training_image or os.environ.get("GODS_MLOPS_TRAINING_IMAGE", ""))
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    compiler.Compiler().compile(
        pipeline_func=build_pipeline(image),
        package_path=str(output),
    )
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=os.environ.get("GODS_MLOPS_TRAINING_IMAGE"))
    parser.add_argument("--output", type=Path, default=Path("gods-mlops-training.yaml"))
    args = parser.parse_args(argv)
    compile_pipeline(args.output, training_image=args.image)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
