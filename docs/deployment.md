# Reproducible training environment

The training image uses Python 3.12, stable PyTorch 2.12.0 with CUDA 13.0 wheels, Transformers 4.57.1, `pycocotools` for detector evaluation, and the pinned model revisions in `models/lock.json`. `uv.lock` captures the complete Python dependency graph and package hashes. The image build installs only from that lock and uses digest-pinned Python and uv base images.

`infra/versions.lock.yaml` records the agreed platform versions and digest-pinned owned images. The storage image is upstream SeaweedFS 4.47 from the official `ghcr.io/chrislusf/seaweedfs` publisher. The upstream project's main license is Apache-2.0; the image also contains separately licensed static assets, so retain its bundled notices. It provides the approved S3-compatible object interface, and upstream Docker instructions document keyless cosign signatures for release images. Label Studio is pinned to the 1.23.2 Community Edition image and upstream Apache-2.0 source. The recorded OCI digests are multi-architecture index digests; Docker selects the platform manifest while preserving the exact index identity.

Validate the lock formats before building:

```bash
uv run gods-mlops check-locks
uv run pytest tests/test_locks.py -q
```

Build and check the training image's model-class and evaluation imports:

```bash
docker build -f images/training/Dockerfile -t gods-mlops/training:0.1.0 .
docker run --rm --network none gods-mlops/training:0.1.0 check-training-image
docker run --rm --gpus all --network none gods-mlops/training:0.1.0 check-training-image --require-cuda
```

The image import check proves that the pinned classes and evaluation modules load. The CUDA form also requires a visible GPU. Neither check loads full model weights or proves model-specific memory use, training convergence, or a completed evaluation.

## Prepare and verify model files

Mount a persistent, writable cache at `/mnt/model-cache`. The preparation command downloads each model and its processor files at the same immutable Hugging Face commit, into a unique directory under `.staging`. It hashes every required file and publishes the revision directory with an atomic rename only after all checks pass. A failed or incomplete download is removed from staging and never becomes a ready cache entry.

```bash
docker run --rm \
  -v /data/jayn2u/gods-mlops/cache:/mnt/model-cache \
  gods-mlops/training:0.1.0 prepare-models

docker run --rm --network none \
  -v /data/jayn2u/gods-mlops/cache:/mnt/model-cache:ro \
  gods-mlops/training:0.1.0 check-models
```

Preparation requires network access to Hugging Face. `check-models` requires every locked processor/configuration and weight file to exist as a regular local file and match its expected SHA-256. A missing file, mutable revision, unsafe path, symlinked artifact, or digest mismatch fails closed. The `check-locks` command validates lock structure only; it never reports the files ready.

The small JSON/tokenizer/processor hashes were computed from files served at the immutable revisions in `models/lock.json`. Large model blobs use the SHA-256 values published in Hugging Face's LFS metadata for those same commits; the preparation command checks the actual downloaded bytes against them. This checkout contains no prepared model cache. The initial lock is not evidence that the models have been downloaded or that the A6000 can train them.

## Image pin sources

- SeaweedFS 4.47 image and signature instructions: [upstream Docker guide](https://github.com/seaweedfs/seaweedfs/blob/4.47/docker/README.md); license: [upstream LICENSE](https://github.com/seaweedfs/seaweedfs/blob/4.47/LICENSE).
- Label Studio 1.23.2 image and license: [upstream release source](https://github.com/HumanSignal/label-studio/tree/1.23.2).
- Training base images: [official Python image](https://hub.docker.com/_/python) and [Astral uv image](https://github.com/astral-sh/uv/pkgs/container/uv).
- Model revisions and published LFS hashes: [RT-DETR API](https://huggingface.co/api/models/PekingU/rtdetr_v2_r18vd?revision=5650961749fa93567c0d46fc7f43ea4f9e914107&blobs=true), [CLIP API](https://huggingface.co/api/models/openai/clip-vit-base-patch16?revision=57c216476eefef5ab752ec549e440a49ae4ae5f3&blobs=true), and [Qwen API](https://huggingface.co/api/models/Qwen/Qwen2.5-VL-7B-Instruct?revision=cc594898137f460bfe9f0759e9844b3ce807cfb5&blobs=true).
