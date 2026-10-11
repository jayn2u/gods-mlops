"""Pre-publication draft pipeline with an explicit human-review boundary."""

import argparse
import os
from pathlib import Path

from kfp import compiler, dsl

from .components import configure_controller_task, controller_component, validate_training_image


def build_pipeline(training_image: str):
    _train_component, prepare_component = controller_component(training_image)

    @dsl.pipeline(name="gods-mlops-annotation-preparation")
    def prepare_pipeline(source_selections_json: str, model_kind: str, config_version: str):
        task = prepare_component(
            source_selections_json=source_selections_json,
            model_kind=model_kind,
            config_version=config_version,
        )
        configure_controller_task(task, training_image=training_image, preparation=True)
        task.set_display_name("GPU draft then Task 5 review assignment")

    return prepare_pipeline


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
    parser.add_argument("--output", type=Path, default=Path("gods-mlops-preparation.yaml"))
    args = parser.parse_args(argv)
    compile_pipeline(args.output, training_image=args.image)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
