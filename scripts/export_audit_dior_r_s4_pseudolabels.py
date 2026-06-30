#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import cv2


CLASSES = [
    "airplane",
    "airport",
    "baseballfield",
    "basketballcourt",
    "bridge",
    "chimney",
    "dam",
    "Expressway-Service-area",
    "Expressway-toll-station",
    "golffield",
    "groundtrackfield",
    "harbor",
    "overpass",
    "ship",
    "stadium",
    "storagetank",
    "tenniscourt",
    "trainstation",
    "vehicle",
    "windmill",
]

SCENE_GROUPS = {
    "aviation": {"airplane", "airport"},
    "sports": {"baseballfield", "basketballcourt", "golffield", "groundtrackfield", "stadium", "tenniscourt"},
    "transport": {"bridge", "Expressway-Service-area", "Expressway-toll-station", "overpass", "trainstation", "vehicle"},
    "water_industrial": {"dam", "harbor", "ship", "storagetank"},
    "industrial_landmark": {"chimney", "windmill"},
}
CLASS_TO_GROUP = {name: group for group, names in SCENE_GROUPS.items() for name in names}


@dataclass(frozen=True)
class Box:
    image_id: str
    label: int
    score: float
    poly: tuple[float, ...]
    teacher: str = ""
    votes: int = 1

    @property
    def cls(self) -> str:
        return CLASSES[self.label]


def tensor_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "tensor"):
        value = value.tensor
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "cpu"):
        value = value.cpu().numpy()
    return np.asarray(value)


def xywht_to_poly(row: np.ndarray) -> tuple[float, ...]:
    cx, cy, w, h, theta = [float(x) for x in row[:5]]
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    corners = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    pts = []
    for x, y in corners:
        pts.extend([cx + x * cos_t - y * sin_t, cy + x * sin_t + y * cos_t])
    return tuple(pts)


def as_poly(row: np.ndarray) -> tuple[float, ...]:
    row = np.asarray(row, dtype=np.float64).reshape(-1)
    if row.size >= 8:
        return tuple(float(x) for x in row[:8])
    if row.size == 5:
        return xywht_to_poly(row)
    raise ValueError(f"Unsupported bbox shape {row.shape}")


def polygon_iou(poly_a: tuple[float, ...], poly_b: tuple[float, ...]) -> float:
    points_a = np.asarray(poly_a, dtype=np.float32).reshape(4, 2)
    points_b = np.asarray(poly_b, dtype=np.float32).reshape(4, 2)
    area_a = abs(float(cv2.contourArea(points_a)))
    area_b = abs(float(cv2.contourArea(points_b)))
    if area_a <= 0 or area_b <= 0:
        return 0.0
    try:
        _, inter = cv2.intersectConvexConvex(points_a, points_b)
    except cv2.error:
        return 0.0
    if inter is None:
        return 0.0
    inter_area = abs(float(cv2.contourArea(inter)))
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def get_field(sample: Any, name: str, default: Any = None) -> Any:
    if isinstance(sample, dict):
        return sample.get(name, default)
    return getattr(sample, name, default)


def load_predictions(path: Path, teacher: str, score_thr: float) -> dict[str, list[Box]]:
    with path.open("rb") as f:
        data = pickle.load(f)
    by_image: dict[str, list[Box]] = defaultdict(list)
    for sample in data:
        pred = get_field(sample, "pred_instances")
        metainfo = get_field(sample, "metainfo", {}) or {}
        image_id = get_field(sample, "img_id", None) or metainfo.get("img_id")
        if image_id is None:
            img_path = get_field(sample, "img_path", None) or metainfo.get("img_path", "")
            image_id = Path(str(img_path)).stem
        image_id = str(image_id)
        if pred is None:
            continue
        bboxes = tensor_to_numpy(get_field(pred, "bboxes"))
        labels = tensor_to_numpy(get_field(pred, "labels")).astype(int)
        scores = tensor_to_numpy(get_field(pred, "scores")).astype(float)
        for bbox, label, score in zip(bboxes, labels, scores):
            if score < score_thr or label < 0 or label >= len(CLASSES):
                continue
            by_image[image_id].append(Box(image_id, int(label), float(score), as_poly(bbox), teacher=teacher))
    return by_image


def load_gt(label_dir: Path) -> dict[str, list[Box]]:
    by_image: dict[str, list[Box]] = defaultdict(list)
    class_to_idx = {name: idx for idx, name in enumerate(CLASSES)}
    class_to_idx.update({name.lower(): idx for idx, name in enumerate(CLASSES)})
    for path in sorted(label_dir.glob("*.txt")):
        image_id = path.stem
        for raw in path.read_text().splitlines():
            parts = raw.split()
            if len(parts) < 9 or parts[0].lower() == "imagesource:" or parts[0].lower() == "gsd:":
                continue
            cls = parts[8]
            label = class_to_idx.get(cls, class_to_idx.get(cls.lower()))
            if label is None:
                continue
            poly = tuple(float(x) for x in parts[:8])
            by_image[image_id].append(Box(image_id, label, 1.0, poly))
    return by_image


def flatten(by_image: dict[str, list[Box]]) -> list[Box]:
    return [box for boxes in by_image.values() for box in boxes]


def group_scene_score(image_boxes: list[Box]) -> float:
    if not image_boxes:
        return 1.0
    groups = [CLASS_TO_GROUP.get(box.cls, "other") for box in image_boxes]
    counts = Counter(groups)
    dominant = counts.most_common(1)[0][1]
    return dominant / len(groups)


def confidence_policy(preds: dict[str, list[Box]], min_score: float) -> dict[str, list[Box]]:
    return {img: [box for box in boxes if box.score >= min_score] for img, boxes in preds.items()}


def hierarchy_scene_policy(preds: dict[str, list[Box]], min_score: float, context_score: float) -> dict[str, list[Box]]:
    out: dict[str, list[Box]] = {}
    for image_id, boxes in preds.items():
        high = [box for box in boxes if box.score >= min_score]
        if not high:
            out[image_id] = []
            continue
        group_counts = Counter(CLASS_TO_GROUP.get(box.cls, "other") for box in high)
        allowed = {group for group, count in group_counts.items() if count / len(high) >= context_score}
        if not allowed:
            allowed = {group_counts.most_common(1)[0][0]}
        out[image_id] = [box for box in high if CLASS_TO_GROUP.get(box.cls, "other") in allowed]
    return out


def agreement_policy(all_preds: dict[str, dict[str, list[Box]]], min_score: float, iou_thr: float) -> dict[str, list[Box]]:
    teachers = list(all_preds)
    image_ids = sorted({img for preds in all_preds.values() for img in preds})
    out: dict[str, list[Box]] = {}
    for image_id in image_ids:
        candidates = []
        for teacher in teachers:
            candidates.extend([box for box in all_preds[teacher].get(image_id, []) if box.score >= min_score])
        candidates.sort(key=lambda box: box.score, reverse=True)
        used: set[tuple[str, int]] = set()
        kept: list[Box] = []
        for cand in candidates:
            if (cand.teacher, id(cand)) in used:
                continue
            cluster = [cand]
            teacher_votes = {cand.teacher}
            for other in candidates:
                if other.teacher in teacher_votes or other.label != cand.label:
                    continue
                if polygon_iou(cand.poly, other.poly) >= iou_thr:
                    cluster.append(other)
                    teacher_votes.add(other.teacher)
            if len(teacher_votes) >= 2:
                best = max(cluster, key=lambda box: box.score)
                avg_score = sum(box.score for box in cluster) / len(cluster)
                kept.append(Box(image_id, best.label, avg_score, best.poly, teacher="agreement", votes=len(teacher_votes)))
                for box in cluster:
                    used.add((box.teacher, id(box)))
        out[image_id] = kept
    return out


def evaluate(preds: dict[str, list[Box]], gt: dict[str, list[Box]], iou_thr: float) -> dict[str, Any]:
    tp = fp = 0
    matched_gt: dict[str, set[int]] = defaultdict(set)
    confusion: Counter[str] = Counter()
    per_class = {name: {"tp": 0, "fp": 0, "gt": 0} for name in CLASSES}
    for boxes in gt.values():
        for box in boxes:
            per_class[box.cls]["gt"] += 1
    for pred in sorted(flatten(preds), key=lambda box: box.score, reverse=True):
        best_iou = 0.0
        best_idx = -1
        best_label = -1
        for idx, target in enumerate(gt.get(pred.image_id, [])):
            iou = polygon_iou(pred.poly, target.poly)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
                best_label = target.label
        if best_iou >= iou_thr and best_idx not in matched_gt[pred.image_id] and best_label == pred.label:
            tp += 1
            per_class[pred.cls]["tp"] += 1
            matched_gt[pred.image_id].add(best_idx)
        else:
            fp += 1
            per_class[pred.cls]["fp"] += 1
            if best_iou >= iou_thr and best_label >= 0 and best_label != pred.label:
                confusion[f"{pred.cls}->{CLASSES[best_label]}"] += 1
    total_gt = sum(len(v) for v in gt.values())
    kept = tp + fp
    precision = tp / kept if kept else 0.0
    recall = tp / total_gt if total_gt else 0.0
    for stats in per_class.values():
        kept_cls = stats["tp"] + stats["fp"]
        stats["precision"] = stats["tp"] / kept_cls if kept_cls else 0.0
        stats["recall"] = stats["tp"] / stats["gt"] if stats["gt"] else 0.0
    scene_scores = [group_scene_score(boxes) for boxes in preds.values() if boxes]
    return {
        "kept_boxes": kept,
        "true_positive": tp,
        "false_positive": fp,
        "gt_boxes": total_gt,
        "precision": precision,
        "recall": recall,
        "class_pair_confusion_top10": confusion.most_common(10),
        "hierarchy_consistency": hierarchy_consistency(preds),
        "scene_consistency_score": float(np.mean(scene_scores)) if scene_scores else 1.0,
        "per_class": per_class,
    }


def hierarchy_consistency(preds: dict[str, list[Box]]) -> float:
    boxes = flatten(preds)
    if not boxes:
        return 1.0
    known = sum(1 for box in boxes if box.cls in CLASS_TO_GROUP)
    return known / len(boxes)


def top_k_like(preds: dict[str, list[Box]], k: int) -> dict[str, list[Box]]:
    selected = sorted(flatten(preds), key=lambda box: box.score, reverse=True)[:k]
    out: dict[str, list[Box]] = defaultdict(list)
    for box in selected:
        out[box.image_id].append(box)
    return dict(out)


def write_dota(policy_dir: Path, preds: dict[str, list[Box]]) -> None:
    label_dir = policy_dir / "labelTxt"
    label_dir.mkdir(parents=True, exist_ok=True)
    for image_id, boxes in preds.items():
        lines = []
        for box in sorted(boxes, key=lambda item: item.score, reverse=True):
            coords = " ".join(f"{coord:.2f}" for coord in box.poly)
            lines.append(f"{coords} {box.cls} 0 {box.score:.6f} votes={box.votes}")
        (label_dir / f"{image_id}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))


def markdown_summary(results: dict[str, Any], gate: dict[str, Any]) -> str:
    lines = ["# DIOR-R S4 Pseudo-Label Pilot Audit", "", "## Quality Table", ""]
    lines.append("| Policy | Kept boxes | Precision | Recall | Hierarchy consistency | Scene consistency | Matched confidence precision |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name, metrics in results["policies"].items():
        matched = metrics.get("matched_confidence_precision")
        matched_s = f"{matched:.6f}" if matched is not None else "n/a"
        lines.append(
            f"| {name} | {metrics['kept_boxes']} | {metrics['precision']:.6f} | {metrics['recall']:.6f} | "
            f"{metrics['hierarchy_consistency']:.6f} | {metrics['scene_consistency_score']:.6f} | {matched_s} |"
        )
    lines.extend(["", "## GeoReason Diagnostic Ladder", ""])
    lines.append("| Level | Diagnostic | Result |")
    lines.append("| --- | --- | --- |")
    lines.append("| R0 | class/prompt grounding quality | proxy: per-class pseudo-label precision/recall in JSON |")
    lines.append("| R1 | confusing-class relation quality | proxy: top class-pair confusion table in JSON |")
    lines.append("| R2 | scene-context consistency | proxy: dominant scene-group consistency score |")
    lines.append("| R3 | final detection/pseudo-label decision quality | gate result below |")
    lines.extend(["", "## Gate", ""])
    for key, value in gate.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Top Class-Pair Confusion", ""])
    for name, metrics in results["policies"].items():
        lines.append(f"- {name}: {metrics['class_pair_confusion_top10']}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export and audit DIOR-R S4 pseudo-label candidates.")
    parser.add_argument("--prediction", action="append", required=True, help="teacher_name=path/to/predictions.pkl")
    parser.add_argument("--gt-label-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--primary-teacher", default=None)
    parser.add_argument("--score-thr", type=float, default=0.30)
    parser.add_argument("--policy-score-thr", type=float, default=0.60)
    parser.add_argument("--context-score", type=float, default=0.55)
    parser.add_argument("--iou-thr", type=float, default=0.50)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gt = load_gt(Path(args.gt_label_dir))
    all_preds: dict[str, dict[str, list[Box]]] = {}
    prediction_paths = {}
    for raw in args.prediction:
        name, path = raw.split("=", 1)
        prediction_paths[name] = str(Path(path).resolve())
        all_preds[name] = load_predictions(Path(path), name, args.score_thr)
    primary = args.primary_teacher or next(iter(all_preds))

    policies = {
        "confidence_only": confidence_policy(all_preds[primary], args.policy_score_thr),
        "hierarchy_scene": hierarchy_scene_policy(all_preds[primary], args.policy_score_thr, args.context_score),
        "teacher_agreement_2of3": agreement_policy(all_preds, args.policy_score_thr, args.iou_thr),
    }

    results: dict[str, Any] = {
        "settings": vars(args),
        "prediction_paths": prediction_paths,
        "primary_teacher": primary,
        "policies": {},
    }
    confidence = policies["confidence_only"]
    for name, preds in policies.items():
        write_dota(output_dir / name, preds)
        metrics = evaluate(preds, gt, args.iou_thr)
        if name != "confidence_only":
            matched = evaluate(top_k_like(confidence, metrics["kept_boxes"]), gt, args.iou_thr)
            metrics["matched_confidence_precision"] = matched["precision"]
            metrics["matched_confidence_recall"] = matched["recall"]
        else:
            metrics["matched_confidence_precision"] = None
            metrics["matched_confidence_recall"] = None
        results["policies"][name] = metrics

    conf = results["policies"]["confidence_only"]
    gate = {
        "precision_improves_at_matched_kept_count": any(
            m.get("matched_confidence_precision") is not None and m["precision"] > m["matched_confidence_precision"]
            for key, m in results["policies"].items()
            if key != "confidence_only"
        ),
        "no_catastrophic_false_positive_expansion": all(
            stats["fp"] <= max(50, 2 * max(1, conf["per_class"][cls]["fp"]))
            for m in results["policies"].values()
            for cls, stats in m["per_class"].items()
        ),
        "usable_recall_without_high_conf_precision_degrade": any(
            m["recall"] > 0 and (m.get("matched_confidence_precision") is None or m["precision"] >= m["matched_confidence_precision"])
            for m in results["policies"].values()
        ),
        "failure_scan_clean": "pending_external_log_scan",
    }
    gate["launch_s4_recommended"] = all(v is True for k, v in gate.items() if k != "failure_scan_clean")
    results["gate"] = gate

    (output_dir / "audit.json").write_text(json.dumps(results, indent=2, sort_keys=True))
    (output_dir / "audit.md").write_text(markdown_summary(results, gate))
    print(json.dumps({"output_dir": str(output_dir), "gate": gate}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
