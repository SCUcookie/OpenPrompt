from __future__ import annotations

from collections import defaultdict
import math
from typing import Any

import cv2
import torch

from openprompt_rs.models.losses import build_supervision_targets


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    relation_matrix: torch.Tensor | None,
    confusing_matrix: torch.Tensor | None,
    device: str,
) -> dict[str, float]:
    model.eval()
    meters: dict[str, float] = defaultdict(float)
    steps = 0
    positive_correct = 0.0
    positive_total = 0.0
    box_l1_total = 0.0

    for batch in dataloader:
        images = batch["images"].to(device)
        targets = batch["targets"]
        outputs = model(images)
        losses = criterion(
            outputs,
            targets,
            relation_matrix=relation_matrix,
            confusing_matrix=confusing_matrix,
        )
        supervision = build_supervision_targets(outputs["query_centers"], targets, outputs["logits"].size(-1))
        predictions = outputs["logits"].argmax(dim=-1)
        mask = supervision["positive_mask"]

        if mask.any():
            positive_correct += (
                predictions[mask] == supervision["label_indices"][mask]
            ).float().sum().item()
            positive_total += float(mask.sum().item())
            box_l1_total += torch.abs(outputs["boxes"][mask] - supervision["box_targets"][mask]).mean().item()

        for key, value in losses.items():
            meters[key] += float(value.item())
        steps += 1

    if steps == 0:
        return {"loss": 0.0, "positive_cls_acc": 0.0, "positive_box_l1": 0.0}

    metrics = {key: value / steps for key, value in meters.items()}
    metrics["positive_cls_acc"] = positive_correct / max(positive_total, 1.0)
    metrics["positive_box_l1"] = box_l1_total / max(steps, 1)
    return metrics


def _cv2_rotated_rect(box: torch.Tensor) -> tuple[tuple[float, float], tuple[float, float], float]:
    cx, cy, width, height, theta = [float(value) for value in box.tolist()]
    return ((cx, cy), (max(width, 1e-6), max(height, 1e-6)), theta * 180.0 / math.pi)


def rotated_box_iou(box_a: torch.Tensor, box_b: torch.Tensor) -> float:
    area_a = float(box_a[2].item() * box_a[3].item())
    area_b = float(box_b[2].item() * box_b[3].item())
    if area_a <= 0.0 or area_b <= 0.0:
        return 0.0

    _, intersection_points = cv2.rotatedRectangleIntersection(
        _cv2_rotated_rect(box_a),
        _cv2_rotated_rect(box_b),
    )
    if intersection_points is None:
        intersection = 0.0
    else:
        hull = cv2.convexHull(intersection_points, returnPoints=True)
        intersection = float(abs(cv2.contourArea(hull)))

    union = area_a + area_b - intersection
    if union <= 0.0:
        return 0.0
    return float(intersection / union)


def _classwise_rotated_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.long)

    keep: list[int] = []
    for class_id in labels.unique(sorted=True).tolist():
        class_indices = torch.nonzero(labels == class_id, as_tuple=False).flatten()
        ordered = class_indices[scores[class_indices].argsort(descending=True)]
        class_keep: list[int] = []
        for candidate in ordered.tolist():
            if all(rotated_box_iou(boxes[candidate], boxes[saved]) < iou_threshold for saved in class_keep):
                class_keep.append(candidate)
        keep.extend(class_keep)

    keep_tensor = torch.tensor(keep, dtype=torch.long)
    if keep_tensor.numel() == 0:
        return keep_tensor
    return keep_tensor[scores[keep_tensor].argsort(descending=True)]


def decode_detections(
    outputs: dict[str, torch.Tensor],
    score_threshold: float = 0.05,
    nms_iou_threshold: float = 0.3,
    max_detections: int = 100,
) -> list[dict[str, torch.Tensor]]:
    probabilities = torch.sigmoid(outputs["logits"])
    scores, labels = probabilities.max(dim=-1)
    detections: list[dict[str, torch.Tensor]] = []

    for batch_idx in range(outputs["boxes"].size(0)):
        batch_scores = scores[batch_idx].detach().cpu()
        batch_labels = labels[batch_idx].detach().cpu()
        batch_boxes = outputs["boxes"][batch_idx].detach().cpu()

        keep = batch_scores >= score_threshold
        batch_scores = batch_scores[keep]
        batch_labels = batch_labels[keep]
        batch_boxes = batch_boxes[keep]
        if batch_scores.numel() == 0:
            detections.append(
                {
                    "boxes": torch.zeros((0, 5), dtype=torch.float32),
                    "labels": torch.zeros((0,), dtype=torch.long),
                    "scores": torch.zeros((0,), dtype=torch.float32),
                }
            )
            continue

        keep_indices = _classwise_rotated_nms(
            boxes=batch_boxes,
            scores=batch_scores,
            labels=batch_labels,
            iou_threshold=nms_iou_threshold,
        )
        if max_detections > 0:
            keep_indices = keep_indices[:max_detections]
        detections.append(
            {
                "boxes": batch_boxes[keep_indices],
                "labels": batch_labels[keep_indices],
                "scores": batch_scores[keep_indices],
            }
        )
    return detections


def _tensor_summary(values: torch.Tensor) -> dict[str, float]:
    if values.numel() == 0:
        return {
            "min": 0.0,
            "p05": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "max": 0.0,
            "mean": 0.0,
        }
    values = values.float().flatten()
    return {
        "min": float(values.min().item()),
        "p05": float(torch.quantile(values, 0.05).item()),
        "p50": float(torch.quantile(values, 0.50).item()),
        "p95": float(torch.quantile(values, 0.95).item()),
        "max": float(values.max().item()),
        "mean": float(values.mean().item()),
    }


def _best_iou_for_detection(
    box: torch.Tensor,
    label: torch.Tensor,
    target: dict[str, torch.Tensor],
    same_class_only: bool,
) -> float:
    target_boxes = target["boxes"].detach().cpu()
    if target_boxes.numel() == 0:
        return 0.0
    target_labels = target["labels"].detach().cpu()
    if same_class_only:
        target_boxes = target_boxes[target_labels == int(label.item())]
    if target_boxes.numel() == 0:
        return 0.0
    return max(rotated_box_iou(box, target_box) for target_box in target_boxes)


@torch.no_grad()
def collect_detection_diagnostics(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    class_names: list[str],
    score_thresholds: list[float] | None = None,
    nms_iou_threshold: float = 0.3,
    max_detections: int = 100,
    max_batches: int | None = None,
) -> dict[str, Any]:
    model.eval()
    thresholds = score_thresholds or [0.05, 0.01, 0.001]
    raw_scores: list[torch.Tensor] = []
    raw_labels: list[torch.Tensor] = []
    raw_boxes: list[torch.Tensor] = []
    gt_class_counts = torch.zeros(len(class_names), dtype=torch.long)
    raw_predicted_class_counts = torch.zeros(len(class_names), dtype=torch.long)
    detections_by_threshold = {
        threshold: {
            "count": 0,
            "per_image": [],
            "scores": [],
            "best_iou_any_class": [],
            "best_iou_same_class": [],
            "class_counts": torch.zeros(len(class_names), dtype=torch.long),
        }
        for threshold in thresholds
    }
    num_images = 0

    for batch_index, batch in enumerate(dataloader):
        if max_batches is not None and batch_index >= max_batches:
            break

        images = batch["images"].to(device)
        outputs = model(images)
        probabilities = torch.sigmoid(outputs["logits"].detach().cpu())
        scores, labels = probabilities.max(dim=-1)
        boxes = outputs["boxes"].detach().cpu()
        raw_scores.append(scores.flatten())
        raw_labels.append(labels.flatten())
        raw_boxes.append(boxes.reshape(-1, boxes.size(-1)))
        raw_predicted_class_counts += torch.bincount(labels.flatten(), minlength=len(class_names))[: len(class_names)]

        for target in batch["targets"]:
            target_labels = target["labels"].detach().cpu()
            if target_labels.numel() > 0:
                gt_class_counts += torch.bincount(target_labels, minlength=len(class_names))[: len(class_names)]

        for threshold in thresholds:
            decoded = decode_detections(
                outputs,
                score_threshold=threshold,
                nms_iou_threshold=nms_iou_threshold,
                max_detections=max_detections,
            )
            threshold_stats = detections_by_threshold[threshold]
            for sample_offset, detection in enumerate(decoded):
                det_count = int(detection["scores"].numel())
                threshold_stats["count"] += det_count
                threshold_stats["per_image"].append(det_count)
                if det_count == 0:
                    continue
                threshold_stats["scores"].extend(float(score.item()) for score in detection["scores"])
                threshold_stats["class_counts"] += torch.bincount(
                    detection["labels"],
                    minlength=len(class_names),
                )[: len(class_names)]
                target = batch["targets"][sample_offset]
                for box, label in zip(detection["boxes"], detection["labels"]):
                    threshold_stats["best_iou_any_class"].append(
                        _best_iou_for_detection(box, label, target, same_class_only=False)
                    )
                    threshold_stats["best_iou_same_class"].append(
                        _best_iou_for_detection(box, label, target, same_class_only=True)
                    )
        num_images += int(images.size(0))

    all_scores = torch.cat(raw_scores) if raw_scores else torch.zeros((0,), dtype=torch.float32)
    all_labels = torch.cat(raw_labels) if raw_labels else torch.zeros((0,), dtype=torch.long)
    all_boxes = torch.cat(raw_boxes) if raw_boxes else torch.zeros((0, 5), dtype=torch.float32)

    threshold_payload: dict[str, Any] = {}
    for threshold, stats in detections_by_threshold.items():
        per_image = torch.tensor(stats["per_image"], dtype=torch.float32)
        score_values = torch.tensor(stats["scores"], dtype=torch.float32)
        any_iou = torch.tensor(stats["best_iou_any_class"], dtype=torch.float32)
        same_class_iou = torch.tensor(stats["best_iou_same_class"], dtype=torch.float32)
        threshold_payload[str(threshold)] = {
            "num_detections": int(stats["count"]),
            "detections_per_image": _tensor_summary(per_image),
            "score_summary": _tensor_summary(score_values),
            "best_iou_any_class_summary": _tensor_summary(any_iou),
            "best_iou_same_class_summary": _tensor_summary(same_class_iou),
            "class_counts": {
                class_name: int(stats["class_counts"][class_id].item())
                for class_id, class_name in enumerate(class_names)
            },
        }

    return {
        "num_eval_images": num_images,
        "num_raw_queries": int(all_scores.numel()),
        "raw_score_summary": _tensor_summary(all_scores),
        "raw_box_summary": {
            "cx": _tensor_summary(all_boxes[:, 0]) if all_boxes.numel() else _tensor_summary(torch.zeros((0,))),
            "cy": _tensor_summary(all_boxes[:, 1]) if all_boxes.numel() else _tensor_summary(torch.zeros((0,))),
            "width": _tensor_summary(all_boxes[:, 2]) if all_boxes.numel() else _tensor_summary(torch.zeros((0,))),
            "height": _tensor_summary(all_boxes[:, 3]) if all_boxes.numel() else _tensor_summary(torch.zeros((0,))),
            "theta": _tensor_summary(all_boxes[:, 4]) if all_boxes.numel() else _tensor_summary(torch.zeros((0,))),
        },
        "raw_predicted_class_counts": {
            class_name: int(raw_predicted_class_counts[class_id].item())
            for class_id, class_name in enumerate(class_names)
        },
        "gt_class_counts": {
            class_name: int(gt_class_counts[class_id].item())
            for class_id, class_name in enumerate(class_names)
        },
        "thresholds": threshold_payload,
        "raw_predicted_unique_classes": int(all_labels.unique().numel()) if all_labels.numel() else 0,
    }


def _average_precision(recalls: torch.Tensor, precisions: torch.Tensor) -> float:
    if recalls.numel() == 0:
        return 0.0
    recall_points = torch.cat([torch.tensor([0.0]), recalls, torch.tensor([1.0])])
    precision_points = torch.cat([torch.tensor([0.0]), precisions, torch.tensor([0.0])])
    for index in range(precision_points.numel() - 1, 0, -1):
        precision_points[index - 1] = torch.maximum(precision_points[index - 1], precision_points[index])
    changes = torch.nonzero(recall_points[1:] != recall_points[:-1], as_tuple=False).flatten()
    return float(((recall_points[changes + 1] - recall_points[changes]) * precision_points[changes + 1]).sum().item())


@torch.no_grad()
def evaluate_detection_map50(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    class_names: list[str],
    score_threshold: float = 0.05,
    nms_iou_threshold: float = 0.3,
    max_detections: int = 100,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    model.eval()

    predictions_by_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
    gt_by_class_and_image: dict[int, dict[int, list[torch.Tensor]]] = defaultdict(lambda: defaultdict(list))
    image_index = 0

    for batch in dataloader:
        images = batch["images"].to(device)
        outputs = model(images)
        detections = decode_detections(
            outputs,
            score_threshold=score_threshold,
            nms_iou_threshold=nms_iou_threshold,
            max_detections=max_detections,
        )

        for sample_offset, target in enumerate(batch["targets"]):
            labels = target["labels"].detach().cpu()
            boxes = target["boxes"].detach().cpu()
            for class_id in range(len(class_names)):
                class_mask = labels == class_id
                if class_mask.any():
                    gt_by_class_and_image[class_id][image_index] = [box for box in boxes[class_mask]]

            detection = detections[sample_offset]
            for box, label, score in zip(detection["boxes"], detection["labels"], detection["scores"]):
                predictions_by_class[int(label.item())].append(
                    {
                        "image_index": image_index,
                        "box": box,
                        "score": float(score.item()),
                    }
                )
            image_index += 1

    metrics: dict[str, Any] = {"num_eval_images": image_index}
    ap_values: list[float] = []
    precision_values: list[float] = []
    recall_values: list[float] = []

    for class_id, class_name in enumerate(class_names):
        class_gt = gt_by_class_and_image[class_id]
        gt_count = sum(len(items) for items in class_gt.values())
        if gt_count == 0:
            continue

        predictions = sorted(predictions_by_class[class_id], key=lambda item: item["score"], reverse=True)
        matched = {
            image_id: [False] * len(image_targets)
            for image_id, image_targets in class_gt.items()
        }
        true_positive: list[float] = []
        false_positive: list[float] = []

        for prediction in predictions:
            image_id = prediction["image_index"]
            targets = class_gt.get(image_id, [])
            best_iou = 0.0
            best_target = -1
            for target_index, target_box in enumerate(targets):
                if matched[image_id][target_index]:
                    continue
                iou = rotated_box_iou(prediction["box"], target_box)
                if iou > best_iou:
                    best_iou = iou
                    best_target = target_index

            if best_iou >= iou_threshold and best_target >= 0:
                matched[image_id][best_target] = True
                true_positive.append(1.0)
                false_positive.append(0.0)
            else:
                true_positive.append(0.0)
                false_positive.append(1.0)

        if true_positive:
            tp_tensor = torch.tensor(true_positive, dtype=torch.float32).cumsum(dim=0)
            fp_tensor = torch.tensor(false_positive, dtype=torch.float32).cumsum(dim=0)
            precisions = tp_tensor / torch.clamp(tp_tensor + fp_tensor, min=1.0)
            recalls = tp_tensor / float(gt_count)
            ap50 = _average_precision(recalls, precisions)
            precision = float(precisions[-1].item())
            recall = float(recalls[-1].item())
        else:
            ap50 = 0.0
            precision = 0.0
            recall = 0.0

        metrics[f"ap50_{class_name}"] = ap50
        metrics[f"precision_{class_name}"] = precision
        metrics[f"recall_{class_name}"] = recall
        ap_values.append(ap50)
        precision_values.append(precision)
        recall_values.append(recall)

    metrics["map50"] = float(sum(ap_values) / max(len(ap_values), 1))
    metrics["mean_precision"] = float(sum(precision_values) / max(len(precision_values), 1))
    metrics["mean_recall"] = float(sum(recall_values) / max(len(recall_values), 1))
    return metrics
