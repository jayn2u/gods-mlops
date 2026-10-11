from __future__ import annotations

import importlib

import pytest

from gods_mlops.training import data


@pytest.fixture
def caption_with_fake_object_store(monkeypatch: pytest.MonkeyPatch):
    store = object()
    calls: list[None] = []

    def make_store():
        calls.append(None)
        return store

    caption = importlib.import_module("gods_mlops.training.caption")
    monkeypatch.setattr(data, "dataset_object_store_from_environment", make_store)
    caption = importlib.reload(caption)
    yield caption, store, calls
    monkeypatch.undo()
    importlib.reload(caption)


def test_object_backed_caption_items_use_the_dataset_object_store(caption_with_fake_object_store):
    caption, store, calls = caption_with_fake_object_store

    selected = caption._object_store_if_needed(
        [{"object": {"key": "datasets/crops/crop-1.jpg", "sha256": "a" * 64, "size_bytes": 12}}]
    )

    assert selected is store
    assert calls == [None]


def test_local_caption_items_do_not_create_an_object_store(caption_with_fake_object_store):
    caption, _, calls = caption_with_fake_object_store

    assert caption._object_store_if_needed([{"image_path": "/tmp/crop-1.jpg"}]) is None
    assert calls == []
