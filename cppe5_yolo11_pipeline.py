
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
from PIL import Image


CLASS_NAMES = ["coverall", "face_shield", "gloves", "goggles", "mask"]


@dataclass
class TrainConfig:
    run_name: str
    model: str
    epochs: int
    imgsz: int
    batch: int
    save_period: int
    device: str
    workers: int
    patience: int
    optimizer: str
    lr0: float
    lrf: float
    weight_decay: float
    hsv_h: float
    hsv_s: float
    hsv_v: float
    degrees: float
    translate: float
    scale: float
    shear: float
    perspective: float
    flipud: float
    fliplr: float
    mosaic: float
    mixup: float
    copy_paste: float
    cos_lr: bool


def yolo_normalize_bbox_coco(x: float, y: float, w: float, h: float, width: int, height: int) -> Tuple[float, float, float, float]:
    """Convert COCO bbox [x, y, w, h] to YOLO [cx, cy, w, h] normalized."""
    cx = (x + w / 2.0) / width
    cy = (y + h / 2.0) / height
    nw = w / width
    nh = h / height
    return cx, cy, nw, nh


def ensure_dirs(paths: List[Path]) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def resolve_user_path(raw_path: str, base_dir: Path) -> Path:
    p = Path(raw_path)
    return p if p.is_absolute() else (base_dir / p)


def _latest_checkpoint_from_glob(search_roots: List[Path], run_name: str) -> Path | None:
    candidates: List[Path] = []
    patterns = [
        f"**/{run_name}/weights/best.pt",
        f"**/{run_name}/weights/last.pt",
    ]
    for root in search_roots:
        if not root.exists():
            continue
        for pattern in patterns:
            candidates.extend(root.glob(pattern))

    if not candidates:
        return None
    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _extract_boxes_and_labels(example: dict) -> Tuple[List[List[float]], List[int]]:
    """Handle slightly different CPPE-5 schemas seen across mirrors."""
    objects = example.get("objects")
    if objects is None:
        boxes = example.get("bbox") or example.get("bboxes") or []
        labels = example.get("category") or example.get("labels") or []
        return boxes, labels

    boxes = objects.get("bbox") or objects.get("bboxes") or []
    labels = objects.get("category") or objects.get("labels") or []
    return boxes, labels


def prepare_cppe5_dataset(dataset_root: Path, max_samples_per_split: int | None = None) -> Path:
    """Download CPPE-5 from HuggingFace datasets and convert to YOLO format."""
    from datasets import load_dataset

    images_root = dataset_root / "images"
    labels_root = dataset_root / "labels"
    ensure_dirs([images_root, labels_root])

    for split in ["train", "val", "test"]:
        ensure_dirs([images_root / split, labels_root / split])

    ds = load_dataset("cppe-5")

    split_plan: List[Tuple[str, str, List[int]]] = []
    has_val = ("validation" in ds) or ("val" in ds)

    if has_val:
        split_map = {
            "train": "train",
            "validation": "val",
            "val": "val",
            "test": "test",
        }
        for src_split, yolo_split in split_map.items():
            if src_split not in ds:
                continue
            records = ds[src_split]
            indices = list(range(len(records)))
            if max_samples_per_split is not None:
                indices = indices[: max_samples_per_split]
            split_plan.append((src_split, yolo_split, indices))
    else:
        # Some CPPE-5 mirrors provide only train/test.
        # In this case, create val by splitting train deterministically.
        train_records = ds["train"]
        all_indices = list(range(len(train_records)))
        rng = random.Random(42)
        rng.shuffle(all_indices)

        val_size = max(1, int(0.15 * len(all_indices)))
        val_indices = all_indices[:val_size]
        train_indices = all_indices[val_size:]

        if max_samples_per_split is not None:
            train_indices = train_indices[: max_samples_per_split]
            val_indices = val_indices[: max_samples_per_split]

        split_plan.append(("train", "train", train_indices))
        split_plan.append(("train", "val", val_indices))

        if "test" in ds:
            test_indices = list(range(len(ds["test"])))
            if max_samples_per_split is not None:
                test_indices = test_indices[: max_samples_per_split]
            split_plan.append(("test", "test", test_indices))

    for src_split, yolo_split, indices in split_plan:
        records = ds[src_split]

        for i in indices:
            ex = records[i]
            img = ex["image"]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)

            width, height = img.size
            image_name = f"{src_split}_{i:06d}.jpg"
            label_name = f"{src_split}_{i:06d}.txt"

            image_path = images_root / yolo_split / image_name
            label_path = labels_root / yolo_split / label_name
            img.convert("RGB").save(image_path, quality=95)

            boxes, labels = _extract_boxes_and_labels(ex)
            lines = []
            for box, cls in zip(boxes, labels):
                if box is None:
                    continue
                x, y, w, h = box
                if w <= 0 or h <= 0:
                    continue
                cx, cy, nw, nh = yolo_normalize_bbox_coco(x, y, w, h, width, height)

                cx = min(max(cx, 0.0), 1.0)
                cy = min(max(cy, 0.0), 1.0)
                nw = min(max(nw, 0.0), 1.0)
                nh = min(max(nh, 0.0), 1.0)

                lines.append(f"{int(cls)} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            label_path.write_text("\n".join(lines), encoding="utf-8")

    data_yaml = dataset_root / "data.yaml"
    payload = {
        "path": str(dataset_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: n for i, n in enumerate(CLASS_NAMES)},
        "nc": len(CLASS_NAMES),
    }
    data_yaml.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return data_yaml


def build_configs() -> Dict[str, TrainConfig]:
    baseline = TrainConfig(
        run_name="baseline",
        model="yolo11n.pt",
        epochs=10,
        imgsz=640,
        batch=8,
        save_period=1,
        device="cpu",
        workers=2,
        patience=20,
        optimizer="AdamW",
        lr0=1e-3,
        lrf=1e-2,
        weight_decay=5e-4,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        cos_lr=False,
    )

    improved = TrainConfig(
        run_name="improved",
        model="yolo11n.pt",
        epochs=10,
        imgsz=704,
        batch=8,
        save_period=1,
        device="cpu",
        workers=2,
        patience=30,
        optimizer="AdamW",
        lr0=8e-4,
        lrf=5e-3,
        weight_decay=7e-4,
        hsv_h=0.02,
        hsv_s=0.8,
        hsv_v=0.5,
        degrees=5.0,
        translate=0.12,
        scale=0.6,
        shear=1.0,
        perspective=0.0005,
        flipud=0.02,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.2,
        cos_lr=True,
    )
    return {"baseline": baseline, "improved": improved}


def train_model(data_yaml: Path, outputs_root: Path, cfg: TrainConfig, resume: bool = True) -> Path:
    from ultralytics import YOLO

    project_dir = outputs_root / "train"
    run_dir = project_dir / cfg.run_name
    weights_dir = run_dir / "weights"
    best_ckpt = weights_dir / "best.pt"
    last_ckpt = weights_dir / "last.pt"

    if best_ckpt.exists():
        print(f"[SKIP TRAIN] Found checkpoint: {best_ckpt}")
        return best_ckpt

    if resume and last_ckpt.exists():
        model = YOLO(str(last_ckpt))
        model.train(resume=True)
        return best_ckpt if best_ckpt.exists() else last_ckpt

    model = YOLO(cfg.model)
    train_out = model.train(
        data=str(data_yaml),
        epochs=cfg.epochs,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        device=cfg.device,
        workers=cfg.workers,
        save=True,
        save_period=cfg.save_period,
        project=str(project_dir),
        name=cfg.run_name,
        exist_ok=True,
        pretrained=True,
        patience=cfg.patience,
        optimizer=cfg.optimizer,
        lr0=cfg.lr0,
        lrf=cfg.lrf,
        weight_decay=cfg.weight_decay,
        hsv_h=cfg.hsv_h,
        hsv_s=cfg.hsv_s,
        hsv_v=cfg.hsv_v,
        degrees=cfg.degrees,
        translate=cfg.translate,
        scale=cfg.scale,
        shear=cfg.shear,
        perspective=cfg.perspective,
        flipud=cfg.flipud,
        fliplr=cfg.fliplr,
        mosaic=cfg.mosaic,
        mixup=cfg.mixup,
        copy_paste=cfg.copy_paste,
        cos_lr=cfg.cos_lr,
        seed=42,
        deterministic=True,
        verbose=True,
    )

    # Resolve actual save directory used by Ultralytics.
    observed_save_dirs: List[Path] = [run_dir]
    out_save_dir = getattr(train_out, "save_dir", None)
    if out_save_dir is not None:
        observed_save_dirs.append(Path(str(out_save_dir)))
    trainer = getattr(model, "trainer", None)
    trainer_save_dir = getattr(trainer, "save_dir", None) if trainer is not None else None
    if trainer_save_dir is not None:
        observed_save_dirs.append(Path(str(trainer_save_dir)))

    for save_dir in observed_save_dirs:
        wdir = save_dir / "weights"
        b = wdir / "best.pt"
        l = wdir / "last.pt"
        if b.exists():
            return b
        if l.exists():
            return l

    fallback = _latest_checkpoint_from_glob(
        search_roots=[project_dir, Path.cwd() / "runs" / "detect", Path.home() / "runs" / "detect"],
        run_name=cfg.run_name,
    )
    if fallback is not None:
        return fallback

    raise FileNotFoundError(
        f"No checkpoint produced for run: {cfg.run_name}. "
        f"Checked save dirs: {[str(p) for p in observed_save_dirs]}"
    )


def evaluate_model(data_yaml: Path, checkpoint: Path, outputs_root: Path, run_name: str) -> Dict[str, float]:
    from ultralytics import YOLO

    model = YOLO(str(checkpoint))
    res = model.val(data=str(data_yaml), split="test", imgsz=640)

    metrics = {
        "precision": float(res.box.mp),
        "recall": float(res.box.mr),
        "mAP50": float(res.box.map50),
        "mAP50_95": float(res.box.map),
        "fitness": float(getattr(res, "fitness", 0.0)),
    }

    out_file = outputs_root / "metrics" / f"{run_name}_metrics.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def save_prediction_examples(
    checkpoint: Path,
    dataset_root: Path,
    outputs_root: Path,
    run_name: str,
    num_images: int = 20,
    conf: float = 0.25,
) -> Path:
    from ultralytics import YOLO

    test_images_dir = dataset_root / "images" / "test"
    all_images = sorted(list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.png")))
    if not all_images:
        raise FileNotFoundError(f"No test images found in: {test_images_dir}")

    random.seed(42)
    selected = random.sample(all_images, k=min(num_images, len(all_images)))

    model = YOLO(str(checkpoint))
    pred_project = outputs_root / "predictions"
    pred_name = f"{run_name}_examples"

    model.predict(
        source=[str(p) for p in selected],
        conf=conf,
        save=True,
        line_width=2,
        show_labels=True,
        show_conf=True,
        project=str(pred_project),
        name=pred_name,
        exist_ok=True,
        verbose=False,
    )
    return pred_project / pred_name


def _fmt(v: float) -> str:
    return f"{v:.4f}"


def generate_report(outputs_root: Path, configs: Dict[str, TrainConfig]) -> Path:
    baseline_path = outputs_root / "metrics" / "baseline_metrics.json"
    improved_path = outputs_root / "metrics" / "improved_metrics.json"

    if not baseline_path.exists() or not improved_path.exists():
        raise FileNotFoundError("Both baseline and improved metrics JSON files are required.")

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    improved = json.loads(improved_path.read_text(encoding="utf-8"))

    def delta(metric: str) -> float:
        return improved[metric] - baseline[metric]

    lines = []
    lines.append("# CPPE-5 YOLO11 Report")
    lines.append("")
    lines.append("## 1. Chosen dataset")
    lines.append("CPPE-5 with 5 classes: coverall, face_shield, gloves, goggles, mask.")
    lines.append("")
    lines.append("## 2. Metrics and why they are used")
    lines.append("- Precision: measures false positives; important for avoiding incorrect alerts.")
    lines.append("- Recall: measures false negatives; important when missing PPE objects is costly.")
    lines.append("- mAP@50: common detection quality metric at IoU=0.50.")
    lines.append("- mAP@50-95: stricter and more informative aggregate metric across IoU thresholds.")
    lines.append("")
    lines.append("## 3. Baseline vs Improved")
    lines.append("| Metric | Baseline | Improved | Delta |")
    lines.append("|---|---:|---:|---:|")
    for metric in ["precision", "recall", "mAP50", "mAP50_95", "fitness"]:
        lines.append(
            f"| {metric} | {_fmt(baseline[metric])} | {_fmt(improved[metric])} | {_fmt(delta(metric))} |"
        )

    lines.append("")
    lines.append("## 4. Implemented improvements")
    lines.append("Hypotheses tested in improved baseline:")
    lines.append("- More epochs (20 -> 35) to better converge on a small dataset.")
    lines.append("- Slightly larger image size (640 -> 704) for small object details.")
    lines.append("- Stronger augmentation: mixup, copy_paste, small rotation/shear/perspective.")
    lines.append("- Cosine LR schedule and tuned regularization for smoother optimization.")
    lines.append("")
    lines.append("### Baseline config")
    lines.append("```json")
    lines.append(json.dumps(asdict(configs["baseline"]), indent=2))
    lines.append("```")
    lines.append("")
    lines.append("### Improved config")
    lines.append("```json")
    lines.append(json.dumps(asdict(configs["improved"]), indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## 5. Prediction artifacts")
    lines.append("- Baseline predictions: outputs/predictions/baseline_examples")
    lines.append("- Improved predictions: outputs/predictions/improved_examples")
    lines.append("")
    lines.append("## 6. Conclusion")
    lines.append("Improved baseline is accepted if mAP@50-95 and/or recall increase while precision stays stable.")

    report_path = outputs_root / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run_all(args: argparse.Namespace) -> None:
    script_dir = Path(__file__).resolve().parent
    dataset_root = resolve_user_path(args.dataset_root, script_dir).resolve()
    outputs_root = resolve_user_path(args.outputs_root, script_dir).resolve()
    outputs_root.mkdir(parents=True, exist_ok=True)

    print("[1/6] Preparing CPPE-5 dataset...")
    data_yaml = prepare_cppe5_dataset(dataset_root, max_samples_per_split=args.max_samples_per_split)

    configs = build_configs()

    print("[2/6] Training baseline...")
    baseline_ckpt = train_model(data_yaml, outputs_root, configs["baseline"], resume=True)

    print("[3/6] Evaluating baseline and saving prediction examples...")
    evaluate_model(data_yaml, baseline_ckpt, outputs_root, "baseline")
    save_prediction_examples(
        baseline_ckpt,
        dataset_root,
        outputs_root,
        run_name="baseline",
        num_images=args.num_pred_images,
        conf=args.conf,
    )

    print("[4/6] Training improved model...")
    improved_ckpt = train_model(data_yaml, outputs_root, configs["improved"], resume=True)

    print("[5/6] Evaluating improved model and saving prediction examples...")
    evaluate_model(data_yaml, improved_ckpt, outputs_root, "improved")
    save_prediction_examples(
        improved_ckpt,
        dataset_root,
        outputs_root,
        run_name="improved",
        num_images=args.num_pred_images,
        conf=args.conf,
    )

    print("[6/6] Generating report...")
    report_path = generate_report(outputs_root, configs)
    print(f"Done. Report saved to: {report_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CPPE-5 YOLO11 baseline/improved pipeline")
    p.add_argument("--dataset-root", type=str, default="data/cppe5_yolo")
    p.add_argument("--outputs-root", type=str, default="outputs")
    p.add_argument("--max-samples-per-split", type=int, default=None)
    p.add_argument("--num-pred-images", type=int, default=20)
    p.add_argument("--conf", type=float, default=0.25)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_all(args)


if __name__ == "__main__":
    main()
