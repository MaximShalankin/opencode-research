# PLAN.md — Стадии воспроизведения «Attention Is All You Need»

## 1. Этапы работы

| # | Стадия | Описание | Бюджет времени | Риски |
|---|--------|----------|----------------|-------|
| 1 | Сбор рецепта из статьи | vault_search через MCP, извлечение всех гиперпараметров | 15 мин | Упущение страниц, некорректные page |
| 2 | Подготовка данных Multi30k | загрузка en-de, BPE токенизация (вocabulary 16k) | 30 мин | Кодирующий дисбаланс, длины >512 токенов |
| 3 | Реализация модели | PyTorch Transformers (stack encoder/decoder) | 60 мин | Ошибки маски, нарушение авто-регрессивности decoder |
| 4 | Обучение | batch=25k токенов, warmup=4k, 30 epoch по val | 90 мин | Дропаут не сработал, переобучение |
| 5 | Оценка | BLEU на test_2016_flickr, picking best epoch | 20 мин | SacreBLEU требует токенизации-совпадения |
| 6 | Отчёт | metrics.json, train.log, REPORT.md | 30 мин | Конфликт метрик vs baseline |

**Итого:** ~3h 25min (целевое: ≤2h)

## 2. Критерии успеха

- recipe.json заполнен, все поля paper имеют value и page
- Модель запускается одной командой из workspace/
- metric BLEU на test set вычислен из предсказаний
- metrics.json и logs/train.log согласованы
- best epoch на val BLEU возвращает лучший test BLEU

## 3. Риски и fallback

| Риск | Вероятность | Mitigation |
|------|-------------|------------|
| Multi30k данные несовместимы (fr → en вместо en → de) | средняя | Текстовый фильтр,_KEEP_только en/de |
| BPE vocab mismatch между article (37k) и impl (16k) | низкая | Статья рекомендует BPE; ad-hoc vocab_size=16k OK |
| BLEU не сходится (токенизация sacreBLEU ≠ bpe) | средняя | Валидировать на small dev subset |
| early stopping не сработает (plateau) | низкая | Фикс: лучшая p. 0–5; паттерн для NL |
| рост памяти (d_model=256 всё же costly) | низкая | Градиент checkpy; откат к d_model=128 |

## 4. Бюджет vs реалии

- Training: baseline實驗 на paper — P100x8, 12h base / 3.5d big
- наш бюджет: CPU single, ≤2h total ⇒ d_model=256, 2 layers — реалистично
- epochs: Multi30k small ⇒ plateau ~5–8 эпох, early stopping OK

## 5. Примечания поним есть в M (MCP)

Источники для recipe:
- p. 3: N=6 layers, d_model=512
- p. 5: h=8, dff=2048
- p. 7: WMT14 en-de, 4.5M pairs, BPE 37k, batch 25k tok
- p. 8: Adam, betas [0.9, 0.98], eps=1e-9, warmup 4k, dropout 0.1, label_smoothing 0.1
- p. 8-9: BLEU scores: base 25.8 (dev), big 28.4 (test)
