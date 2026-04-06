from __future__ import annotations

import math

from PIL import Image

from openprompt_rs.data.dota import DotaOBBDataset


def _write_rgb_image(path: str, size: tuple[int, int] = (100, 80)) -> None:
    Image.new("RGB", size=size, color=(128, 128, 128)).save(path)


def test_dota_dataset_parses_original_format(tmpdir) -> None:
    image_dir = tmpdir.mkdir("images")
    label_dir = tmpdir.mkdir("labels")

    image_path = image_dir.join("sample.png")
    label_path = label_dir.join("sample.txt")
    _write_rgb_image(str(image_path))
    label_path.write(
        "10 20 30 20 30 40 10 40 plane 0\n",
    )

    dataset = DotaOBBDataset(
        image_dir=str(image_dir),
        label_dir=str(label_dir),
        class_names=["plane", "ship"],
        image_size=64,
    )
    sample = dataset[0]

    assert sample["target"]["boxes"].shape == (1, 5)
    assert sample["target"]["labels"].tolist() == [0]


def test_dota_dataset_parses_numeric_normalized_format(tmpdir) -> None:
    image_dir = tmpdir.mkdir("images")
    label_dir = tmpdir.mkdir("labels")

    image_path = image_dir.join("sample.png")
    label_path = label_dir.join("sample.txt")
    _write_rgb_image(str(image_path))
    label_path.write(
        "1 0.10 0.20 0.30 0.20 0.30 0.50 0.10 0.50\n",
    )

    dataset = DotaOBBDataset(
        image_dir=str(image_dir),
        label_dir=str(label_dir),
        class_names=["plane", "ship"],
        image_size=64,
    )
    sample = dataset[0]

    assert sample["target"]["boxes"].shape == (1, 5)
    assert sample["target"]["labels"].tolist() == [1]
    cx, cy, width, height, angle = sample["target"]["boxes"][0].tolist()
    assert 0.19 < cx < 0.21
    assert 0.34 < cy < 0.36
    assert 0.23 < width < 0.25
    assert 0.24 < height < 0.26
    assert abs(angle - (math.pi / 2.0)) < 1e-6


def test_dota_dataset_builds_positive_tiles(tmpdir) -> None:
    image_dir = tmpdir.mkdir("images")
    label_dir = tmpdir.mkdir("labels")

    image_path = image_dir.join("sample.png")
    label_path = label_dir.join("sample.txt")
    _write_rgb_image(str(image_path), size=(100, 100))
    label_path.write(
        "0 0.10 0.10 0.30 0.10 0.30 0.30 0.10 0.30\n"
        "1 0.60 0.60 0.80 0.60 0.80 0.80 0.60 0.80\n",
    )

    dataset = DotaOBBDataset(
        image_dir=str(image_dir),
        label_dir=str(label_dir),
        class_names=["plane", "ship"],
        image_size=64,
        tile_size=50,
        tile_stride=50,
        include_empty_tiles=False,
    )

    assert len(dataset) == 2
    first = dataset[0]
    second = dataset[1]
    assert first["target"]["labels"].tolist() == [0]
    assert second["target"]["labels"].tolist() == [1]
    assert first["meta"]["tile_object_count"] == 1
    assert second["meta"]["tile_object_count"] == 1


def test_dota_dataset_keeps_non_overlapping_tail_tiles(tmpdir) -> None:
    image_dir = tmpdir.mkdir("images")
    label_dir = tmpdir.mkdir("labels")

    image_path = image_dir.join("sample.png")
    _write_rgb_image(str(image_path), size=(100, 40))

    dataset = DotaOBBDataset(
        image_dir=str(image_dir),
        label_dir=str(label_dir),
        class_names=["plane", "ship"],
        image_size=64,
        tile_size=40,
        tile_stride=40,
        include_empty_tiles=True,
    )

    assert len(dataset) == 3
    assert [sample["meta"]["tile_x"] for sample in dataset] == [0, 40, 80]
    assert [sample["meta"]["tile_w"] for sample in dataset] == [40, 40, 20]


def test_dota_dataset_assigns_overlap_objects_once(tmpdir) -> None:
    image_dir = tmpdir.mkdir("images")
    label_dir = tmpdir.mkdir("labels")

    image_path = image_dir.join("sample.png")
    label_path = label_dir.join("sample.txt")
    _write_rgb_image(str(image_path), size=(100, 40))
    label_path.write(
        "0 0.10 0.10 0.30 0.10 0.30 0.30 0.10 0.30\n"
        "1 0.50 0.10 0.60 0.10 0.60 0.30 0.50 0.30\n"
        "0 0.80 0.10 0.90 0.10 0.90 0.30 0.80 0.30\n",
    )

    dataset = DotaOBBDataset(
        image_dir=str(image_dir),
        label_dir=str(label_dir),
        class_names=["plane", "ship"],
        image_size=64,
        tile_size=60,
        tile_stride=40,
        include_empty_tiles=False,
    )

    assert len(dataset) == 2
    assert sum(sample["target"]["labels"].numel() for sample in dataset) == 3
    assert dataset[0]["target"]["labels"].tolist() == [0]
    assert sorted(dataset[1]["target"]["labels"].tolist()) == [0, 1]
