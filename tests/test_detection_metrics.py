from __future__ import annotations

import torch

from openprompt_rs.engine.evaluator import decode_detections, evaluate_detection_map50, rotated_box_iou


class FixedModel(torch.nn.Module):
    def __init__(self, outputs: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.outputs = outputs

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.outputs


def test_rotated_box_iou_identity() -> None:
    box = torch.tensor([0.5, 0.5, 0.2, 0.1, 0.0], dtype=torch.float32)
    assert abs(rotated_box_iou(box, box) - 1.0) < 1e-6


def test_decode_detections_filters_by_score() -> None:
    outputs = {
        "logits": torch.tensor([[[8.0, -8.0], [-8.0, -8.0]]], dtype=torch.float32),
        "boxes": torch.tensor([[[0.5, 0.5, 0.2, 0.1, 0.0], [0.3, 0.3, 0.1, 0.1, 0.0]]], dtype=torch.float32),
    }
    detections = decode_detections(outputs, score_threshold=0.5, nms_iou_threshold=0.3, max_detections=10)
    assert len(detections) == 1
    assert detections[0]["boxes"].shape == (1, 5)
    assert detections[0]["labels"].tolist() == [0]


def test_evaluate_detection_map50_perfect_prediction() -> None:
    model = FixedModel(
        outputs={
            "logits": torch.tensor([[[10.0, -10.0]]], dtype=torch.float32),
            "boxes": torch.tensor([[[0.5, 0.5, 0.2, 0.1, 0.0]]], dtype=torch.float32),
        }
    )
    dataloader = [
        {
            "images": torch.zeros((1, 3, 32, 32), dtype=torch.float32),
            "targets": [
                {
                    "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.1, 0.0]], dtype=torch.float32),
                    "labels": torch.tensor([0], dtype=torch.long),
                }
            ],
        }
    ]
    metrics = evaluate_detection_map50(
        model=model,
        dataloader=dataloader,
        device="cpu",
        class_names=["plane", "ship"],
        score_threshold=0.05,
        nms_iou_threshold=0.3,
        max_detections=10,
    )
    assert abs(metrics["ap50_plane"] - 1.0) < 1e-6
    assert abs(metrics["map50"] - 1.0) < 1e-6
