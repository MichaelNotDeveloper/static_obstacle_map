# Static Obstacle Map

Проект строит бинарную карту статических препятствий в формате Bird's Eye View (BEV) по изображениям четырех автомобильных камер.

На вход модели поступают:

- четыре RGB-изображения;
- intrinsic-матрица каждой камеры;
- матрица `car_to_cam` каждой камеры;
- опционально карта глубины для каждого изображения.

На выходе получается тензор размера `(B, 1, 188, 126)`. После применения `sigmoid` и порога он преобразуется в `int32`-карту из значений `0` и `1`.

## Архитектура

Основной используемый режим запуска:

```text
4 RGB cameras
      |
      v
проекция пикселей на плоскость земли
      |
      v
12 RGB BEV-каналов + 1 coverage-канал
      |
      v
BEVSegNet
      |
      v
occupancy logits: (B, 1, 188, 126)
```

В этом режиме используются аргументы `--no_depth --no_feature_model`.

### Датасет

Класс `BaseDataset` находится в `src/dataset.py`.

Датасет:

1. Читает пути из `info.csv`.
2. Загружает изображения четырех камер.
3. Изменяет их размер до `256 x 512`.
4. Для train применяет слабый `ColorJitter` по яркости, контрасту, насыщенности и оттенку.
5. Нормализует RGB по ImageNet mean/std.
6. Загружает intrinsic- и `car_to_cam`-матрицы.
7. В train/val дополнительно загружает GT-карту `(1, 188, 126)`.

Порядок камер фиксирован:

```text
0: /camera/inner/frontal/middle
1: /camera/inner/frontal/far
2: /side/left/forward
3: /side/right/forward
```

Для масштабирования intrinsics используются исходные размеры изображений каждой камеры:

```text
frontal/middle: 546 x 1024
frontal/far:    568 x 1024
side/left:      540 x 1024
side/right:     540 x 1024
```

`collate_camera_batch` возвращает список из четырех батчей изображений и такие же списки depth, intrinsics и extrinsics:

```text
images[camera_id]:     (B, 3, 256, 512)
depths[camera_id]:     (B, 1, 256, 512)
intrinsics[camera_id]: (B, 3, 4)
car2cams[camera_id]:   (B, 4, 4)
target:                (B, 1, 188, 126)
```

### Проекция в BEV

Проектор `DepthToBEVProjection` находится в `src/bev_projection.py` и является `nn.Module`.

В основном режиме без depth каждый пиксель задает луч камеры:

```text
x_ray = (u - cx) / fx
y_ray = (v - cy) / fy
ray_cam = [x_ray, y_ray, 1]
```

Матрица `car_to_cam` инвертируется в `cam_to_car`. Луч и положение камеры переводятся в систему координат автомобиля. Затем находится пересечение луча с плоскостью земли `z = ground_z`:

```text
t = (ground_z - camera_origin_z) / ray_car_z
point_car = camera_origin + t * ray_car
```

Точка автомобиля переводится в координаты BEV:

```text
row = (x_car - x_min) / meters_per_pixel
col = (bev_y_sign * y_car - y_min) / meters_per_pixel
```

Значение пикселя распределяется по четырем ближайшим BEV-ячейкам с bilinear-весами. Каналы камер разделены, поэтому основной режим формирует:

```text
camera 0 RGB: channels 0..2
camera 1 RGB: channels 3..5
camera 2 RGB: channels 6..8
camera 3 RGB: channels 9..11
coverage:     channel 12
```

Каждый RGB-блок нормализуется только весами своей камеры. Coverage содержит долю камер, покрывающих конкретную BEV-ячейку: `0`, `0.25`, `0.5`, `0.75` или `1.0`.

Размер BEV равен `(188, 126)`, разрешение примерно `150 / 188 = 0.798` метра на пиксель.

### Опциональные модели

Проект сохраняет два дополнительных режима:

- `--use_feature_model`: `RoadPixelFeatureNet` строит обучаемый вектор признаков каждого пикселя до проекции. По умолчанию признаки камер хранятся раздельно, поэтому вход BEV-модели имеет `4 * feature_dim + 1` канал.
- `--use_depth`: Depth Anything V2 Metric Outdoor предсказывает глубину в метрах, после чего пиксели разворачиваются в 3D и переводятся в систему координат автомобиля.

В основном скрипте оба режима отключены, поэтому модель Depth Anything не загружается.

### BEVSegNet

`BEVSegNet` находится в `src/obstacle_model.py`. Это компактная residual U-Net:

- stem с `Conv2d`, `GroupNorm`, `SiLU` и residual-блоком;
- три downsampling-уровня;
- context-блок с dilation `1`, `2` и `4`;
- residual bottleneck;
- три upsampling-уровня со skip connections;
- выходная свертка в один канал logits.

При `base_ch_map=16` модель имеет около одного миллиона параметров.

## Обучение

Используется сумма двух функций потерь:

```text
loss = bce_weight * BCEWithLogits + dice_weight * DiceLoss
```

Пиксели GT со значением `ignore_index` исключаются из обеих частей loss и из метрик. Для датасета конкурса используется `ignore_index=255`.

Оптимизатор: `AdamW`.

Learning rate меняется после каждого batch:

1. Linear warmup от `min_lr` к базовому LR.
2. Cosine decay от базового LR обратно к `min_lr`.

Если `--warmup_steps=0`, число шагов прогрева вычисляется как:

```text
warmup_epochs * len(train_dataloader)
```

Для CUDA включается mixed precision через `torch.amp.autocast` и `GradScaler`. В tqdm показываются текущие loss, IoU, LR и норма градиента.

После каждой эпохи logger:

- выводит train/val метрики;
- записывает историю в `runs/train/metrics.json`;
- сохраняет график в `runs/train/plots/`;
- сохраняет `runs/train/checkpoints/best.pt`, если улучшился `val_iou`.

## Структура данных

`training.py` ожидает следующую структуру относительно `--data_dir`:

```text
DATA_DIR/
├── train/
│   └── autonomy_yandex_dataset_train/
│       ├── info.csv
│       ├── images/
│       ├── matrices/
│       └── static_grids/
├── val/
│   └── autonomy_yandex_dataset_val/
│       ├── info.csv
│       ├── images/
│       ├── matrices/
│       └── static_grids/
└── test/
    └── autonomy_yandex_dataset_test/
        ├── info.csv
        ├── images/
        └── matrices/
```

Путь к test можно передать отдельно через `--test_data_dir`.

Пути внутри `info.csv` разрешаются относительно корня соответствующего датасета. Абсолютные пути также поддерживаются.

## Зависимости

Для основного режима нужны:

```text
numpy
pandas
Pillow
matplotlib
tqdm
torch
torchvision
```

Пакет `transformers` нужен только при запуске с `--use_depth`.

## Запуск

Перед запуском на Kaggle необходимо заменить `--data_dir` и `--test_data_dir` в `run_training.sh` на пути из `/kaggle/input/...`.

Основной запуск:

```bash
bash run_training.sh
```

Эквивалентная сокращенная команда:

```bash
python3 training.py \
  --data_dir /kaggle/input/your-dataset \
  --test_data_dir /kaggle/input/your-dataset/test/autonomy_yandex_dataset_test \
  --epochs 10 \
  --batch_size 4 \
  --mapping_lr 1e-4 \
  --min_lr 1e-5 \
  --warmup_epochs 2 \
  --pixel_step 1 \
  --bce_weight 1.0 \
  --dice_weight 1.0 \
  --augment \
  --use_coverage_map \
  --no_depth \
  --no_feature_model \
  --mixed_precision
```

Для обучения без автоматического прохода по test добавьте:

```bash
--skip_test
```

## Генерация submission

Если `--skip_test` не указан, после последней эпохи код:

1. Загружает checkpoint с лучшим `val_iou`.
2. Прогоняет test dataset.
3. Применяет `sigmoid` и `--pred_threshold`.
4. Сохраняет бинарные `int32`-матрицы в:

```text
submission/
├── info.csv
└── predicted_static_grids/
    ├── ...npy
    └── ...npy
```

Перед отправкой следует проверить, что количество `.npy` совпадает с количеством строк test `info.csv`, каждая матрица имеет форму `(188, 126)`, тип `int32` и содержит только `0` и `1`.

## Основные параметры

| Аргумент | Назначение |
|---|---|
| `--epochs` | Число эпох обучения |
| `--batch_size` | Размер batch |
| `--pixel_step` | Шаг выборки пикселей перед проекцией |
| `--mapping_lr` | LR модели BEV-карты |
| `--feat_lr` | LR feature model |
| `--min_lr` | Минимальный LR для warmup/cosine |
| `--warmup_steps` | Явное число batch-шагов warmup |
| `--warmup_epochs` | Число эпох warmup, если `warmup_steps=0` |
| `--weight_decay` | Weight decay оптимизатора AdamW |
| `--bce_weight` | Вес BCE в loss |
| `--dice_weight` | Вес Dice в loss |
| `--pred_threshold` | Порог бинаризации результата |
| `--base_ch_map` | Базовая ширина BEVSegNet |
| `--feature_dim` | Размер вектора pixel features |
| `--num_workers` | Число DataLoader workers |
| `--use_coverage_map` | Добавить coverage-канал |
| `--augment` | Включить слабые RGB-аугментации train |
| `--no_depth` | Использовать проекцию на плоскость земли без depth model |
| `--no_feature_model` | Проецировать RGB напрямую |
| `--skip_test` | Не запускать test после обучения |
