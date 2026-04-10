from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import box_iou, nms
from torchvision.transforms import functional as TF

try:
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "torchmetrics is required. Install with: pip install torchmetrics"
    ) from exc


CLASS_NAMES = ["coverall", "face_shield", "gloves", "goggles", "mask"]
NUM_CLASSES = len(CLASS_NAMES)


@dataclass
class TrainState:
    epoch: int = 0
    best_map: float = -1.0


class YOLOTxtDataset(Dataset):
    def __init__(self, data_root: Path, split: str, img_size: int):
        self.data_root = data_root
        self.split = split
        self.img_size = img_size
        self.images_dir = data_root / "images" / split
        self.labels_dir = data_root / "labels" / split
        self.image_paths = sorted(list(self.images_dir.glob("*.jpg")) + list(self.images_dir.glob("*.png")))
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {self.images_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    @staticmethod
    def _read_label_file(label_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        if not label_path.exists() or label_path.stat().st_size == 0:
            return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.long)

        boxes = []
        labels = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id = int(float(parts[0]))
            cx, cy, w, h = map(float, parts[1:])
            x1 = max(0.0, cx - w / 2.0)
            y1 = max(0.0, cy - h / 2.0)
            x2 = min(1.0, cx + w / 2.0)
            y2 = min(1.0, cy + h / 2.0)
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(cls_id)

        if not boxes:
            return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.long)

        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        label_path = self.labels_dir / (img_path.stem + ".txt")

        image = Image.open(img_path).convert("RGB")
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        image_t = TF.to_tensor(image)

        boxes, labels = self._read_label_file(label_path)
        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([idx], dtype=torch.long)}
        return image_t, target, img_path.name


def collate_fn(batch):
    images = torch.stack([x[0] for x in batch], dim=0)
    targets = [x[1] for x in batch]
    names = [x[2] for x in batch]
    return images, targets, names


class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int = 3, s: int = 1):
        super().__init__()
        p = k // 2
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, k, s, p, bias=False),
            nn.BatchNorm2d(c_out),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class TinyGridDetector(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = nn.Sequential(
            ConvBlock(3, 16, 3, 2),
            ConvBlock(16, 32, 3, 2),
            ConvBlock(32, 64, 3, 2),
            ConvBlock(64, 128, 3, 2),
            ConvBlock(128, 192, 3, 2),
            ConvBlock(192, 192, 3, 1),
        )
        self.head = nn.Conv2d(192, 5 + num_classes, 1, 1, 0)

    def forward(self, x):
        feat = self.backbone(x)
        out = self.head(feat)  # [B, 5+C, S, S]
        return out


def decode_predictions(raw_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode model output to bbox+obj+class logits per cell.

    Returns:
      boxes_xyxy: [B, S, S, 4] in normalized coordinates
      obj_logits: [B, S, S]
      cls_logits: [B, S, S, C]
    """
    b, ch, s, _ = raw_out.shape
    obj_logits = raw_out[:, 0, :, :]

    tx = raw_out[:, 1, :, :]
    ty = raw_out[:, 2, :, :]
    tw = raw_out[:, 3, :, :]
    th = raw_out[:, 4, :, :]
    cls_logits = raw_out[:, 5:, :, :].permute(0, 2, 3, 1).contiguous()

    device = raw_out.device
    ys = torch.arange(s, device=device).view(1, s, 1).expand(b, s, s)
    xs = torch.arange(s, device=device).view(1, 1, s).expand(b, s, s)

    cx = (xs + torch.sigmoid(tx)) / s
    cy = (ys + torch.sigmoid(ty)) / s
    w = torch.sigmoid(tw)
    h = torch.sigmoid(th)

    x1 = (cx - w / 2.0).clamp(0.0, 1.0)
    y1 = (cy - h / 2.0).clamp(0.0, 1.0)
    x2 = (cx + w / 2.0).clamp(0.0, 1.0)
    y2 = (cy + h / 2.0).clamp(0.0, 1.0)

    boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)
    return boxes_xyxy, obj_logits, cls_logits


def build_targets(targets: List[Dict[str, torch.Tensor]], grid_size: int, device: torch.device):
    b = len(targets)
    obj_t = torch.zeros((b, grid_size, grid_size), dtype=torch.float32, device=device)
    box_t = torch.zeros((b, grid_size, grid_size, 4), dtype=torch.float32, device=device)  # cx,cy,w,h
    cls_t = torch.full((b, grid_size, grid_size), -1, dtype=torch.long, device=device)

    for bi, t in enumerate(targets):
        boxes = t["boxes"].to(device)
        labels = t["labels"].to(device)
        if boxes.numel() == 0:
            continue

        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        order = torch.argsort(areas, descending=True)

        for idx in order:
            x1, y1, x2, y2 = boxes[idx]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            w = (x2 - x1).clamp(min=1e-6)
            h = (y2 - y1).clamp(min=1e-6)

            gx = min(grid_size - 1, max(0, int((cx * grid_size).item())))
            gy = min(grid_size - 1, max(0, int((cy * grid_size).item())))

            if obj_t[bi, gy, gx] > 0.5:
                continue

            obj_t[bi, gy, gx] = 1.0
            box_t[bi, gy, gx, :] = torch.tensor([cx, cy, w, h], device=device)
            cls_t[bi, gy, gx] = labels[idx]

    return obj_t, box_t, cls_t


def compute_loss(raw_out: torch.Tensor, targets: List[Dict[str, torch.Tensor]], lambda_box=5.0, lambda_cls=1.0):
    boxes_xyxy, obj_logits, cls_logits = decode_predictions(raw_out)
    b, s, _, _ = boxes_xyxy.shape
    device = raw_out.device

    obj_t, box_t, cls_t = build_targets(targets, grid_size=s, device=device)

    # objectness loss over all cells
    obj_loss = F.binary_cross_entropy_with_logits(obj_logits, obj_t)

    # box loss only on positive cells
    pos_mask = obj_t > 0.5
    if pos_mask.any():
        pred_boxes = boxes_xyxy[pos_mask]  # xyxy
        # Convert predicted to cxcywh for stable regression against target encoding
        px = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2.0
        py = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2.0
        pw = (pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(min=1e-6)
        ph = (pred_boxes[:, 3] - pred_boxes[:, 1]).clamp(min=1e-6)
        pred_cxcywh = torch.stack([px, py, pw, ph], dim=1)

        tgt_cxcywh = box_t[pos_mask]
        box_loss = F.smooth_l1_loss(pred_cxcywh, tgt_cxcywh)

        cls_pred = cls_logits[pos_mask]
        cls_gt = cls_t[pos_mask]
        cls_loss = F.cross_entropy(cls_pred, cls_gt)
    else:
        box_loss = torch.tensor(0.0, device=device)
        cls_loss = torch.tensor(0.0, device=device)

    total = obj_loss + lambda_box * box_loss + lambda_cls * cls_loss
    return total, {"obj": obj_loss.item(), "box": box_loss.item(), "cls": cls_loss.item()}


@torch.no_grad()
def predict_batch(
    model: nn.Module,
    images: torch.Tensor,
    conf_thres: float = 0.25,
    nms_iou: float = 0.5,
):
    raw = model(images)
    boxes_xyxy, obj_logits, cls_logits = decode_predictions(raw)

    b, s, _ = obj_logits.shape
    out_preds = []
    obj_scores = torch.sigmoid(obj_logits)
    cls_probs = torch.softmax(cls_logits, dim=-1)

    for i in range(b):
        obj = obj_scores[i].reshape(-1)
        cls_p = cls_probs[i].reshape(-1, NUM_CLASSES)
        boxes = boxes_xyxy[i].reshape(-1, 4)

        cls_conf, cls_idx = torch.max(cls_p, dim=1)
        scores = obj * cls_conf

        keep = scores >= conf_thres
        boxes = boxes[keep]
        scores = scores[keep]
        labels = cls_idx[keep]

        if boxes.numel() > 0:
            keep_nms = nms(boxes, scores, nms_iou)
            boxes = boxes[keep_nms]
            scores = scores[keep_nms]
            labels = labels[keep_nms]
        else:
            boxes = torch.zeros((0, 4), device=images.device)
            scores = torch.zeros((0,), device=images.device)
            labels = torch.zeros((0,), dtype=torch.long, device=images.device)

        out_preds.append({"boxes": boxes, "scores": scores, "labels": labels})

    return out_preds


def _precision_recall_iou50(
    preds: List[Dict[str, torch.Tensor]], targets: List[Dict[str, torch.Tensor]], iou_thr: float = 0.5
) -> Tuple[float, float]:
    tp = 0
    fp = 0
    fn = 0

    for p, t in zip(preds, targets):
        p_boxes = p["boxes"].detach().cpu()
        p_labels = p["labels"].detach().cpu()
        p_scores = p["scores"].detach().cpu()

        t_boxes = t["boxes"].detach().cpu()
        t_labels = t["labels"].detach().cpu()

        if p_boxes.numel() == 0 and t_boxes.numel() == 0:
            continue
        if p_boxes.numel() == 0:
            fn += int(t_boxes.shape[0])
            continue
        if t_boxes.numel() == 0:
            fp += int(p_boxes.shape[0])
            continue

        order = torch.argsort(p_scores, descending=True)
        p_boxes = p_boxes[order]
        p_labels = p_labels[order]

        matched_gt = torch.zeros((t_boxes.shape[0],), dtype=torch.bool)

        for bi in range(p_boxes.shape[0]):
            pred_box = p_boxes[bi : bi + 1]
            pred_lbl = p_labels[bi]
            same_cls = torch.where(t_labels == pred_lbl)[0]
            if same_cls.numel() == 0:
                fp += 1
                continue

            candidate_idx = same_cls[~matched_gt[same_cls]]
            if candidate_idx.numel() == 0:
                fp += 1
                continue

            ious = box_iou(pred_box, t_boxes[candidate_idx]).squeeze(0)
            best_iou, best_pos = torch.max(ious, dim=0)
            if best_iou.item() >= iou_thr:
                gt_idx = candidate_idx[best_pos]
                matched_gt[gt_idx] = True
                tp += 1
            else:
                fp += 1

        fn += int((~matched_gt).sum().item())

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    return precision, recall


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    conf_thres: float,
    nms_iou: float,
) -> Dict[str, float]:
    model.eval()
    map_metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")

    all_preds = []
    all_targets = []

    for images, targets, _names in loader:
        images = images.to(device)
        preds = predict_batch(model, images, conf_thres=conf_thres, nms_iou=nms_iou)

        metric_preds = []
        metric_targets = []
        for i in range(len(preds)):
            metric_preds.append(
                {
                    "boxes": preds[i]["boxes"].detach().cpu(),
                    "scores": preds[i]["scores"].detach().cpu(),
                    "labels": preds[i]["labels"].detach().cpu(),
                }
            )
            metric_targets.append(
                {
                    "boxes": targets[i]["boxes"].detach().cpu(),
                    "labels": targets[i]["labels"].detach().cpu(),
                }
            )

        map_metric.update(metric_preds, metric_targets)
        all_preds.extend(metric_preds)
        all_targets.extend(metric_targets)

    map_out = map_metric.compute()
    precision, recall = _precision_recall_iou50(all_preds, all_targets, iou_thr=0.5)

    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "mAP50": float(map_out["map_50"].item()),
        "mAP50_95": float(map_out["map"].item()),
        "fitness": float(map_out["map"].item()),
    }
    return metrics


@torch.no_grad()
def save_prediction_examples(
    model: nn.Module,
    dataset: YOLOTxtDataset,
    device: torch.device,
    out_dir: Path,
    num_images: int,
    conf_thres: float,
    nms_iou: float,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    indices = list(range(len(dataset)))
    random.Random(42).shuffle(indices)
    indices = indices[: min(num_images, len(indices))]

    model.eval()
    for idx in indices:
        image_t, _target, img_name = dataset[idx]
        image = image_t.unsqueeze(0).to(device)
        preds = predict_batch(model, image, conf_thres=conf_thres, nms_iou=nms_iou)[0]

        img_pil = TF.to_pil_image(image_t)
        draw = ImageDraw.Draw(img_pil)

        for box, score, label in zip(preds["boxes"].cpu(), preds["scores"].cpu(), preds["labels"].cpu()):
            x1 = int(box[0].item() * img_pil.width)
            y1 = int(box[1].item() * img_pil.height)
            x2 = int(box[2].item() * img_pil.width)
            y2 = int(box[3].item() * img_pil.height)
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
            label_text = f"{CLASS_NAMES[int(label.item())]} {score.item():.2f}"
            draw.text((x1 + 2, max(0, y1 - 12)), label_text, fill=(255, 0, 0))

        img_pil.save(out_dir / img_name)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    state: TrainState,
    ckpt_path: Path,
):
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": state.epoch,
            "best_map": state.best_map,
        },
        ckpt_path,
    )


def load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, ckpt_path: Path) -> TrainState:
    data = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(data["model"])
    optimizer.load_state_dict(data["optimizer"])
    return TrainState(epoch=int(data.get("epoch", 0)), best_map=float(data.get("best_map", -1.0)))


def generate_report(metrics: Dict[str, float], out_path: Path) -> None:
    lines = [
        "# Отчет по собственной модели детекции (CPPE-5)",
        "",
        "## Выбранные метрики",
        "- Precision: доля корректных срабатываний среди всех предсказаний.",
        "- Recall: доля найденных объектов среди всех реальных объектов.",
        "- mAP@50: качество детекции при IoU=0.5.",
        "- mAP@50-95: усредненная строгая метрика по диапазону IoU.",
        "",
        "## Результаты на тестовой выборке",
        f"- precision: {metrics['precision']:.4f}",
        f"- recall: {metrics['recall']:.4f}",
        f"- mAP50: {metrics['mAP50']:.4f}",
        f"- mAP50_95: {metrics['mAP50_95']:.4f}",
        f"- fitness: {metrics['fitness']:.4f}",
        "",
        "## Артефакты",
        "- Чекпоинты: outputs/custom_detector/checkpoints",
        "- Предсказания: outputs/custom_detector/predictions",
        "- Метрики JSON: outputs/custom_detector/metrics/custom_metrics.json",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def run(args: argparse.Namespace):
    script_dir = Path(__file__).resolve().parent
    data_root = (script_dir / args.data_root).resolve() if not Path(args.data_root).is_absolute() else Path(args.data_root)
    outputs_root = (script_dir / args.outputs_root).resolve() if not Path(args.outputs_root).is_absolute() else Path(args.outputs_root)

    if not (data_root / "images").exists() or not (data_root / "labels").exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_root}. Run cppe5_yolo11_pipeline.py once to prepare dataset."
        )

    ckpt_dir = outputs_root / "checkpoints"
    pred_dir = outputs_root / "predictions"
    metrics_dir = outputs_root / "metrics"

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    train_ds = YOLOTxtDataset(data_root, "train", img_size=args.img_size)
    val_ds = YOLOTxtDataset(data_root, "val", img_size=args.img_size)
    test_ds = YOLOTxtDataset(data_root, "test", img_size=args.img_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    model = TinyGridDetector(num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    state = TrainState(epoch=0, best_map=-1.0)
    last_ckpt = ckpt_dir / "last.pt"
    best_ckpt = ckpt_dir / "best.pt"

    if args.resume and last_ckpt.exists():
        state = load_checkpoint(model, optimizer, last_ckpt)
        print(f"Resumed from {last_ckpt} at epoch={state.epoch}, best_map={state.best_map:.4f}")

    for epoch in range(state.epoch + 1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0

        for images, targets, _ in train_loader:
            images = images.to(device)

            optimizer.zero_grad(set_to_none=True)
            raw_out = model(images)
            loss, _parts = compute_loss(raw_out, targets, lambda_box=args.lambda_box, lambda_cls=args.lambda_cls)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += float(loss.item())
            steps += 1

        avg_loss = total_loss / max(1, steps)

        val_metrics = evaluate(
            model,
            val_loader,
            device=device,
            conf_thres=args.conf_thres,
            nms_iou=args.nms_iou,
        )

        state.epoch = epoch
        save_checkpoint(model, optimizer, state, last_ckpt)
        save_checkpoint(model, optimizer, state, ckpt_dir / f"epoch_{epoch:03d}.pt")

        if val_metrics["mAP50_95"] > state.best_map:
            state.best_map = val_metrics["mAP50_95"]
            save_checkpoint(model, optimizer, state, best_ckpt)

        print(
            f"Epoch {epoch}/{args.epochs} | loss={avg_loss:.4f} | "
            f"val mAP50={val_metrics['mAP50']:.4f} | val mAP50-95={val_metrics['mAP50_95']:.4f}"
        )

    # Final evaluation on test using best checkpoint if available
    if best_ckpt.exists():
        final_state = torch.load(best_ckpt, map_location="cpu")
        model.load_state_dict(final_state["model"])

    test_metrics = evaluate(
        model,
        test_loader,
        device=device,
        conf_thres=args.conf_thres,
        nms_iou=args.nms_iou,
    )

    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "custom_metrics.json"
    metrics_path.write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")

    save_prediction_examples(
        model,
        test_ds,
        device=device,
        out_dir=pred_dir,
        num_images=args.num_pred_images,
        conf_thres=args.conf_thres,
        nms_iou=args.nms_iou,
    )

    report_path = outputs_root / "report_custom_detector.md"
    generate_report(test_metrics, report_path)

    print(f"Done. Metrics: {metrics_path}")
    print(f"Done. Predictions: {pred_dir}")
    print(f"Done. Report: {report_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Custom CPPE-5 detector training and evaluation")
    p.add_argument("--data-root", type=str, default="data/cppe5_yolo")
    p.add_argument("--outputs-root", type=str, default="outputs/custom_detector")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lambda-box", type=float, default=5.0)
    p.add_argument("--lambda-cls", type=float, default=1.0)
    p.add_argument("--conf-thres", type=float, default=0.25)
    p.add_argument("--nms-iou", type=float, default=0.5)
    p.add_argument("--num-pred-images", type=int, default=20)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, cuda:0")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    run(args)


if __name__ == "__main__":
    main()
