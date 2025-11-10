# Восстановление решений уравнений Навье-Стокса методом физически-информированного обучения с избыточными физическими ограничениями

**Short description (EN):** Reproducible notebooks, datasets and pretrained checkpoints for reconstructing 2D Taylor–Green flow fields from sparse/noisy observations using physics-informed training. Includes data for 32×32, 64×64, 128×128 grids and the accompanying paper.

> Подробное описание постановки, метрик и результатов см. в [`paper.pdf`](./paper.pdf).

---

## Содержание
- [Структура репозитория](#структура-репозитория)
- [Данные](#данные)
- [Установка](#установка)
- [Быстрый старт](#быстрый-старт)
- [Чекпоинты](#чекпоинты)
- [Оценка и визуализация](#оценка-и-визуализация)
- [Воспроизводимость](#воспроизводимость)
- [Цитирование](#цитирование)
- [Лицензия](#лицензия)
- [Контакты](#контакты)

---

## Структура репозитория

~~~text
│   paper.pdf
│   tgMLP_128x128_a1.ipynb
│   tgMLP_128x128_a1_a2.ipynb
│   tgMLP_128x128_full.ipynb
│   tgMLP_32x32_a1.ipynb
│   tgMLP_32x32_a1_a2.ipynb
│   tgMLP_32x32_full.ipynb
│   tgMLP_64x64_a1.ipynb
│   tgMLP_64x64_a1_a2.ipynb
│   tgMLP_64x64_full.ipynb
│   tg_data.ipynb
│
├───checkpoints
│       tgMLP_128x128_epoch=200.pth
│       tgMLP_128x128_epoch=200_a1.pth
│       tgMLP_128x128_epoch=200_a1_a2.pth
│       tgMLP_32x32_epoch=200.pth
│       tgMLP_32x32_epoch=200_a1.pth
│       tgMLP_32x32_epoch=a1_a2.pth
│       tgMLP_64x64_epoch=200.pth
│       tgMLP_64x64_epoch=200_a1.pth
│       tgMLP_64x64_epoch=200_a1_a2.pth
│
└───data
    ├───128x128
    │   ├───clean
    │   │       E_clean_TG_Re6.0_NX128_NY128_NT101.npz
    │   │       E_clean_TG_Re6.0_NX128_NY128_NT101_anim.gif
    │   ├───SNR=10
    │   │       eTG_NX128_NY128_NT101_SNR10.npz
    │   │       eTG_NX128_NY128_NT101_SNR10_anim.gif
    │   ├───SNR=20
    │   │       eTG_NX128_NY128_NT101_SNR20.npz
    │   │       eTG_NX128_NY128_NT101_SNR20_anim.gif
    │   └───SNR=30
    │           eTG_NX128_NY128_NT101_SNR30.npz
    │           eTG_NX128_NY128_NT101_SNR30_anim.gif
    │
    ├───32x32
    │   ├───clean
    │   │       E_clean_TG_Re6.0_NX32_NY32_NT101.npz
    │   │       E_clean_TG_Re6.0_NX32_NY32_NT101_anim.gif
    │   ├───SNR=10
    │   │       eTG_NX32_NY32_NT101_SNR10.npz
    │   │       eTG_NX32_NY32_NT101_SNR10_anim.gif
    │   ├───SNR=20
    │   │       eTG_NX32_NY32_NT101_SNR20.npz
    │   │       eTG_NX32_NY32_NT101_SNR20_anim.gif
    │   └───SNR=30
    │           eTG_NX32_NY32_NT101_SNR30.npz
    │           eTG_NX32_NY32_NT101_SNR30_anim.gif
    │
    └───64x64
        ├───clean
        │       E_clean_TG_Re6.0_NX64_NY64_NT101.npz
        │       E_clean_TG_Re6.0_NX64_NY64_NT101_anim.gif
        │
        ├───SNR=10
        │       eTG_NX64_NY64_NT101_SNR10.npz
        │       eTG_NX64_NY64_NT101_SNR10_anim.gif
        │
        ├───SNR=20
        │       eTG_NX64_NY64_NT101_SNR20.npz
        │       eTG_NX64_NY64_NT101_SNR20_anim.gif
        │
        └───SNR=30
                eTG_NX64_NY64_NT101_SNR30.npz
                eTG_NX64_NY64_NT101_SNR30_anim.gif
~~~

Названия ноутбуков отражают состав ограничений:
- `*_a1` — согласование с наблюдениями.
- `*_a1_a2` — наблюдения + дифференциальные невязки уравнений.
- `*_full` — полный состав ограничений (см. детали в `paper.pdf`).

---

## Данные

Используется классическое течение Тейлора–Грина при фиксированном числе Рейнольдса (см. `paper.pdf`), периодические граничные условия, сетки `32×32`, `64×64`, `128×128`, временные ряды длины `NT=101`. Для каждой сетки доступны:
- `clean` — «чистые» эталонные поля;
- `SNR=10/20/30` — зашумлённые наблюдения с указанным отношением сигнал/шум.

Файлы `*_anim.gif` визуализируют временную эволюцию полей.

---

## Установка

Требуется **Python 3.10+**.

~~~bash
# Рекомендуется отдельное окружение
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Базовые зависимости для запуска ноутбуков
pip install numpy scipy matplotlib jupyter tqdm

# PyTorch установите согласно вашей платформе (CPU/GPU), пример для CPU:
pip install torch
~~~

> GPU не обязателен, но существенно ускоряет обучение; ноутбуки автоматически используют CUDA при её наличии.

---

## Быстрый старт

1. Откройте ноутбук под нужную сетку и конфигурацию, например `tgMLP_64x64_full.ipynb`.
2. В ячейке с путями к данным выберите источник:
   - `./data/<NXxNY>/clean/*.npz` или один из наборов `SNR=*`.
3. Запустите все ячейки. По завершении обучения модель и метрики сохраняются в `./checkpoints`.
4. Для экспресс-проверки без обучения загрузите соответствующий чекпоинт (см. раздел ниже) и выполните ячейки оценки.

---

## Чекпоинты

В каталоге `checkpoints/` размещены готовые веса для разных сеток и составов потерь:

~~~text
tgMLP_128x128_epoch=200.pth
tgMLP_128x128_epoch=200_a1.pth
tgMLP_128x128_epoch=200_a1_a2.pth
tgMLP_32x32_epoch=200.pth
tgMLP_32x32_epoch=200_a1.pth
tgMLP_32x32_epoch=a1_a2.pth
tgMLP_64x64_epoch=200.pth
tgMLP_64x64_epoch=200_a1.pth
tgMLP_64x64_epoch=200_a1_a2.pth
~~~

Пример загрузки весов в PyTorch (вне ноутбука):

~~~python
import torch

# Инициализируйте модель с теми же гиперпараметрами, что и в ноутбуке
model = ...

# Пример: загрузка полного сценария на 64×64
state = torch.load("checkpoints/tgMLP_64x64_epoch=200.pth", map_location="cpu")
model.load_state_dict(state)
model.eval()
~~~

---

## Оценка и визуализация

- В ноутбуках предусмотрены ячейки для расчёта метрик (например, MSE и относительных ошибок по компонентам поля).
- Визуализации временной эволюции доступны в `data/**/clean` и `data/**/SNR=*` через файлы `*_anim.gif`.
- Точные определения метрик, нормировок и процедуры сопоставления с эталоном изложены в `paper.pdf`.

---

## Воспроизводимость

- Ноутбуки фиксируют ключевые гиперпараметры и генераторы случайных чисел.
- Для сравнения сценариев используйте одинаковые источники данных и шаги предобработки.
- Подробная методология и допущения описаны в `paper.pdf`.

---

## Цитирование

Если вы используете этот код, данные или результаты, пожалуйста, цитируйте:

~~~text
[Paper] Хомяков Д.В., Кудинов В.А., Восстановление решений уравнений Навье-Стокса методом физически-информированного обучения с избыточными физическими ограничениями ... [в работе]
~~~

Черновик BibTeX:

~~~bibtex
[в работе]
@misc{tg_pinn_2025,
  title        = {Восстановление решений уравнений Навье-Стокса методом физически-информированного обучения с избыточными физическими ограничениями},
  year         = {2025},
  note         = {Code, data and checkpoints available in the repository},
  howpublished = {}}
}
~~~

---

## Лицензия

`MIT`

---

## Контакты

Вопросы и предложения — через Issues репозитория, либо по почте khomyakov_dv@kursksu.ru
