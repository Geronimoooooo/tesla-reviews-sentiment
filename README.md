# Tesla Reviews Sentiment — бинарная классификация / binary classification

> **RU.** Распознавание позитивных и негативных отзывов владельцев
> автомобилей Tesla нейросетью `Embedding → GlobalAveragePooling →
> Dense`.
>
> **EN.** Positive vs. negative sentiment classification of Tesla
> owner reviews with an `Embedding → GlobalAveragePooling → Dense`
> network.

---

## Описание / Overview

**RU.** База — два текстовых файла с позитивными и негативными
отзывами. Полный NLP-пайплайн: явное чтение файлов по именам
классов, анализ символьного состава, разбиение 80/20, балансировка
классов обрезкой длинного класса до длины короткого, удаление
эмодзи регулярным выражением, токенизация с `VOCAB_SIZE=10000`,
скользящее окно по 500 слов с шагом 50, обучение сети `Embedding →
SpatialDropout1D → GlobalAveragePooling1D → BatchNorm → Dense →
Dropout → Dense`.

**EN.** The dataset consists of two text files — positive and
negative reviews. The full NLP pipeline: explicit file reading by
class name, per-character analysis, 80/20 split, class balancing by
truncating the longer class to match the shorter one, emoji removal
via regex, tokenisation with `VOCAB_SIZE=10000`, 500-word sliding
windows with a stride of 50, and training an `Embedding →
SpatialDropout1D → GlobalAveragePooling1D → BatchNorm → Dense →
Dropout → Dense` network.

## Датасет / Dataset

- **Источник / Source:**
  <https://storage.yandexcloud.net/aiueducation/Content/base/l7/tesla.zip>
- Архив скачивается и распаковывается автоматически в секции 2.
- **Классы / Classes:** `Негативный отзыв`, `Позитивный отзыв`.

## Стек / Stack

- Python 3.11
- `numpy`, `matplotlib`, `gdown`
- `scikit-learn` (`confusion_matrix`, `ConfusionMatrixDisplay`)
- `tensorflow` / `keras` (`Embedding`, `GlobalAveragePooling1D`,
  `SpatialDropout1D`, `BatchNormalization`, `Dense`, `Dropout`,
  `Tokenizer`)

## Структура / Structure

```
tesla-reviews-sentiment/
├── README.md
└── tesla_reviews_sentiment.ipynb
```

Логические разделы / notebook sections:

1. Импорты / Imports
2. Загрузка и распаковка / Download and unzip
3. Чтение файлов и предпросмотр / Read and preview
4. Статистика по символам / Per-character statistics
5. Разбиение 80 / 20
6. Балансировка классов / Class balancing
7. Удаление эмодзи / Emoji removal
8. Токенизация / Tokenisation
9. Нарезка скользящим окном / Sliding-window slicing
10. Helpers: `plot_history`, `evaluate_classifier`,
    `compile_train_evaluate`
11. Модель `Embedding → GAP → Dense`
12. Выводы / Conclusions

## Результаты / Results

**RU.**

- Модель уверенно достигает целевых 85–90 % точности на валидации.
- Балансировка классов по длине — самый важный приём: без неё сеть
  просто «угадывает» преобладающий класс.
- Регуляризация `SpatialDropout1D(0.2)` + `Dropout(0.2)` +
  `BatchNorm` позволяет держать сеть далеко от переобучения на
  компактном датасете.

**EN.**

- The model comfortably reaches the 85–90 % accuracy target on the
  validation set.
- Length-based class balancing is the single most important trick;
  without it the network simply predicts the dominant class.
- Regularisation via `SpatialDropout1D(0.2)` + `Dropout(0.2)` +
  `BatchNorm` keeps the network far from overfitting on a compact
  dataset.

## Как запустить / How to run

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install numpy matplotlib gdown scikit-learn tensorflow jupyter

jupyter notebook tesla_reviews_sentiment.ipynb
```

**RU.** Архив (~0.5 МБ) скачивается автоматически в первой секции
ноутбука.

**EN.** The archive (~0.5 MB) is downloaded automatically in the
notebook's first section.
