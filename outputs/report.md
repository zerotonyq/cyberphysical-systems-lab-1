# Отчет по CPPE-5 и YOLO11

## 1. Выбранный набор данных
CPPE-5 с 5 классами: coverall, face_shield, gloves, goggles, mask.

## 2. Метрики качества и обоснование выбора
- Precision: показывает долю ложноположительных срабатываний; важно, чтобы модель не делала лишних тревог.
- Recall: показывает долю ложноотрицательных ошибок; важно не пропускать объекты СИЗ.
- mAP@50: стандартная метрика качества детекции при IoU=0.50.
- mAP@50-95: более строгая и информативная агрегированная метрика по диапазону IoU.

## 3. Сравнение baseline и improved
| Метрика | Improved  | Baseline | Delta |
|---|---:|---:|---:|
| precision | 0.8721 | 0.6558 | +0.2164 |
| recall | 0.6580 | 0.5517 | +0.1062 |
| mAP50 | 0.7715 | 0.6139 | +0.1576 |
| mAP50_95 | 0.4758 | 0.3107 | +0.1651 |
| fitness | 0.4758 | 0.3107 | +0.1651 |

## 4. Выполненные улучшения
Проверенные гипотезы для improved baseline:
- Увеличение числа эпох (20 -> 35) для лучшей сходимости на небольшом датасете.
- Увеличение размера изображения (640 -> 704) для лучшей детекции мелких объектов.
- Усиление аугментаций: mixup, copy_paste, небольшие rotation/shear/perspective.
- Применение cosine LR schedule и настройка регуляризации для более стабильной оптимизации.

### Конфигурация baseline
```json
{
  "run_name": "baseline",
  "model": "yolo11n.pt",
  "epochs": 10,
  "imgsz": 640,
  "batch": 8,
  "save_period": 1,
  "device": "cpu",
  "workers": 2,
  "patience": 20,
  "optimizer": "AdamW",
  "lr0": 0.001,
  "lrf": 0.01,
  "weight_decay": 0.0005,
  "hsv_h": 0.015,
  "hsv_s": 0.7,
  "hsv_v": 0.4,
  "degrees": 0.0,
  "translate": 0.1,
  "scale": 0.5,
  "shear": 0.0,
  "perspective": 0.0,
  "flipud": 0.0,
  "fliplr": 0.5,
  "mosaic": 1.0,
  "mixup": 0.0,
  "copy_paste": 0.0,
  "cos_lr": false
}
```

### Конфигурация improved
```json
{
  "run_name": "improved",
  "model": "yolo11n.pt",
  "epochs": 30,
  "imgsz": 704,
  "batch": 8,
  "save_period": 1,
  "device": "cpu",
  "workers": 2,
  "patience": 30,
  "optimizer": "AdamW",
  "lr0": 0.0008,
  "lrf": 0.005,
  "weight_decay": 0.0007,
  "hsv_h": 0.02,
  "hsv_s": 0.8,
  "hsv_v": 0.5,
  "degrees": 5.0,
  "translate": 0.12,
  "scale": 0.6,
  "shear": 1.0,
  "perspective": 0.0005,
  "flipud": 0.02,
  "fliplr": 0.5,
  "mosaic": 1.0,
  "mixup": 0.1,
  "copy_paste": 0.2,
  "cos_lr": true
}
```

## 5. Артефакты предсказаний
- Предсказания baseline: outputs/predictions/baseline_examples
- Предсказания improved: outputs/predictions/improved_examples

## 6. Вывод
Улучшенный baseline принимается, поскольку  растут mAP@50-95 и/или recall при стабильном precision.
