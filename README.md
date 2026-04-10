# CPPE-5: YOLO11 и собственная модель детекции

Проект содержит две части:
- `cppe5_yolo11_pipeline.py` — эксперименты с готовыми моделями Ultralytics YOLO11
- `custom_detector_cppe5.py` — самостоятельная имплементация модели детекции с классификацией

## 1) Установка

```bash
cd project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Часть A: YOLO11 (Ultralytics)

Полный запуск:

```bash
python cppe5_yolo11_pipeline.py
```

Быстрый запуск (меньше данных):

```bash
python cppe5_yolo11_pipeline.py --max-samples-per-split 300
```

Артефакты:
- Чекпоинты: `outputs/train/baseline/weights/`, `outputs/train/improved/weights/`
- Предсказания: `outputs/predictions/baseline_examples/`, `outputs/predictions/improved_examples/`
- Метрики: `outputs/metrics/baseline_metrics.json`, `outputs/metrics/improved_metrics.json`
- Отчет: `outputs/report.md`

## 3) Часть B: Собственная модель детекции

Важно: сначала должен быть подготовлен датасет в формате YOLO (`data/cppe5_yolo`).
Если уже запускался `cppe5_yolo11_pipeline.py`, датасет уже готов.

Запуск обучения и оценки:

```bash
python custom_detector_cppe5.py --epochs 25 --batch-size 16 --img-size 256 --device auto
```

Продолжить обучение с последнего чекпоинта:

```bash
python custom_detector_cppe5.py --epochs 25 --resume --device auto
```

Быстрый тестовый прогон:

```bash
python custom_detector_cppe5.py --epochs 5 --batch-size 8 --img-size 224 --device cpu
```

Артефакты собственной модели:
- Чекпоинты: `outputs/custom_detector/checkpoints/` (`last.pt`, `best.pt`, `epoch_XXX.pt`)
- Предсказания: `outputs/custom_detector/predictions/`
- Метрики: `outputs/custom_detector/metrics/custom_metrics.json`
- Отчет: `outputs/custom_detector/report_custom_detector.md`

## 4) Метрики

Во всех экспериментах используются одинаковые метрики:
- `precision`
- `recall`
- `mAP50`
- `mAP50_95`


Скрипты работают на CPU и GPU.
Для слабого железа нужно уменьшить параметры `--img-size`, `--batch-size` и `--epochs`.
