# REPORT — масштабированное воспроизведение «Attention Is All You Need» (Multi30k en→de)

## Суть

Это **не** воспроизведение WMT14/28.4 BLEU. Это **масштабированный baseline**:
малый Transformer (2 слоя, d_model 256) на Multi30k (29k пар против ~4.5M в
статье), обученный на одном GPU ~105 с. Числа статьи (27.3/28.4 BLEU,
p. 8, Table 2) и наш результат (22.11 BLEU на Multi30k test) — это **разные
постановки** и не сравниваются напрямую; сравниваемость есть только с
локальным эталоном `reference/` (та же постановка, 13.09 BLEU).

Все факты из статьи цитируются со страницами по выдаче MCP
(`vault_search`/`vault_get_note`, `source=papers`, doc `aiayn`).

## Таблица «утверждение → значение → доказательство»

| Утверждение | Значение | Доказательство |
|---|---|---|
| Наш test BLEU (Multi30k test_2016_flickr, sacreBLEU 13a) | 22.11 | `workspace/metrics.json` (test_bleu); `workspace/logs/train.log` строка 39 |
| Наш лучший val BLEU и лучшая эпоха | 23.27 @ epoch 26 | `metrics.json` (val_bleu[25]=23.27); `logs/train.log` строки 33, 38 |
| Наш train_loss (лучшая эпоха) | 2.8831 | `metrics.json`; `logs/train.log` строка 33 |
| Конфиг модели (layers/d_model/heads/d_ff) | 2 / 256 / 4 / 1024 | `metrics.json` config; `logs/train.log` строка 1; `model.py:29` |
| Dropout / label smoothing / warmup | 0.1 / 0.1 / 4000 | `metrics.json`; `logs/train.log` строка 1; `train.py:37` |
| Vocab (BPE, shared en+de) | 16000 | `metrics.json`; `logs/train.log` строка 2; `artifacts/bpe.json` |
| Batch (src+tgt токенов) | 2048 | `metrics.json`; `train.py:118`; `logs/train.log` строки 6, 8 |
| Эпох / total steps | 30 / 12390 | `metrics.json` (epochs=30); `logs/train.log` строки 8–37 |
| Данные (train/val/test пар) | 29000 / 1014 / 1000 | `logs/train.log` строки 3–5 |
| Скорость обучения | ~409–413 шагов/эпоху, 838563 ток/эпоху | `logs/train.log` строки 6–8 |
| Параметры модели | 7.82M | `logs/train.log` строка 7 |
| Устройство и версия torch | cuda, torch 2.12.1+cu130 | `logs/train.log` строка 1 |
| Wall clock / скорость / пик памяти | 105.4 с / 480993 tok/s / 2340.7 MB | `metrics.json` |
| **Эталон** test/val BLEU (та же постановка) | 13.09 / 12.09 (best epoch 5 из 10) | `reference/metrics-reference.json` |
| Статья: WMT14 EN-DE, ~4.5M пар, BPE ~37000 | 4500000 / 37000 | статья p. 7 (§5.1); `recipe.json` paper.data |
| Статья: batch ≈ 25000 src + 25000 tgt токенов | 25000 | статья p. 7; `recipe.json` paper.batch_tokens |
| Статья: Adam β=(0.9, 0.98), ε=1e-9, warmup 4000, Noam (3) | — | статья p. 7 (§5.3, формула (3)); `recipe.json` paper.optimizer |
| Статья: base 6 слоёв, d_model 512; big h=8, d_ff 2048 | 6/512; 8/2048 | статья p. 3, p. 5; `recipe.json` paper.model |
| Статья: Pdrop=0.1, label smoothing εls=0.1 | 0.1 / 0.1 | статья p. 8; `recipe.json` paper.model |
| Статья: BLEU newstest2014, base EN-DE = 27.3, big = 28.4 | 27.3 / 28.4 | статья p. 8 (Table 2); `recipe.json` paper.reported_bleu |
| Статья: beam search (beam 4, α=0.6), усреднение чекпоинтов | — | статья p. 8; `recipe.json` paper.eval |

## Воспроизведено

1. **Архитектура Transformer** (encoder–decoder, multi-head attention,
   positional embeddings, FFN, residual+norm): `model.py:28–97` —
   `nn.TransformerEncoder`/`nn.TransformerDecoder` c 2 слоями, d_model 256,
   4 heads, d_ff 1024 (масштабированные значения от base/big из статьи
   p. 3, p. 5). Тренировка сходится: loss 51.96 → 2.73, val BLEU 0.02 →
   23.27 (`logs/train.log` строки 8–37).
2. **Оптимизатор и расписание LR**: Adam β=(0.9, 0.98), ε=1e-9
   (`train.py:182`) — совпадает со статьёй p. 7. Noam-расписание
   `lrate = d_model^-0.5 * min(step^-0.5, step*warmup^-1.5)`
   (`model.py:100–106`): warmup до 4000 шагов, затем inverse-sqrt затухание.
   Проверка по логу: step 413 → lr 0.000102 (= 0.0625·413·4000^-1.5),
   step 4130 → lr 0.000973 (= 0.0625·4130^-0.5), step 12390 → 0.000561
   (`logs/train.log` строки 8, 17, 37). Та же формула и у эталона
   (`reference/run_reference.py`, noam_lr).
3. **Label smoothing 0.1** (`model.py:10–25`, `train.py:37`) — как в статье
   p. 8; `ignore_index=0` исключает padding.
4. **BPE-токенизатор, общий для en и de** (`bpe.py`, vocab 16000,
   `artifacts/bpe.json`) — аналог shared BPE из статьи p. 7 (~37000).
5. **Оценка sacreBLEU 13a, greedy decode, best checkpoint по val**
   (`train.py:94–96, 219–229, 242–245`): тест считается с лучшей эпохи
   (26) — требование task.md; эталон работает так же
   (`reference/metrics-reference.json`: best epoch 5, test 13.09).
6. **Седирование**: seed 42 для random/numpy/torch (`train.py:33, 40–45`),
   конфиг и версия torch зафиксированы в `logs/train.log` строка 1.

## Не воспроизведено (сознательно, в силу масштабирования)

1. **Данные**: Multi30k 29000 пар vs WMT14 ~4.5M пар (статья p. 7).
   `logs/train.log` строки 3–5.
2. **Масштаб модели**: 2 слоя / d_model 256 / 4 heads / d_ff 1024 (7.82M
   params, `logs/train.log` строка 7) vs base 6/512/8/2048 и big
   (статья p. 3, p. 5). Это ~5–10× меньше параметров.
3. **Объём обучения**: 12390 шагов / 105 с (1 GPU, `logs/train.log`
   строки 37, 40) vs 100k шагов (base) / 300k (big) на 8×P100 (статья p. 7–8).
4. **Batch**: 2048 src+tgt токенов (`train.py:118`) vs ~25000+25000
   (статья p. 7).
5. **Декодирование**: greedy (`model.py:68–97`) vs beam 4 + length penalty
   α=0.6 + усреднение чекпоинтов (статья p. 8).
6. **Целевой результат**: наш 22.11 на Multi30k test ≠ 27.3/28.4 на
   newstest2014 (статья p. 8) — разные тестовые наборы и постановки.
   Сопоставимо только с эталоном: 22.11 vs 13.09
   (`reference/metrics-reference.json`) — наш прогон обучался дольше
   (12390 vs ~4500 шагов) и с другим батчингом, см. ниже.
7. **Отличия реализации от эталона** (`reference/run_reference.py`):
   - Tied embeddings (один `nn.Embedding` для src/tgt, `out_proj` = embed,
     `model.py:35, 44–45`) vs separate src/tgt embeddings + отдельный
     output Linear в эталоне.
   - Learned positional embeddings (`model.py:36–37`) vs sinusoidal
     PositionalEncoding в эталоне.
   - Token-weighted loss по валидным токенам (`model.py:17–25`:
     `loss.sum() / valid_tokens`) vs `CrossEntropyLoss` по всем элементам
     включая padding в эталоне — поэтому loss не сопоставимы по масштабу
     (наш epoch 1: 51.96 vs эталон 7.31; `reference/train-reference.log`).
   - Динамический батчинг по 2048 токенов (`train.py:64–79`) vs фикс. 64
     пары.
   - max_len 64 vs 80 (не влияло: `dropped_long=0` у обоих, `logs/train.log`
     строки 3–5).

## Не проверялось

1. **Воспроизводимость BLEU 27.3/28.4** (статья p. 8): нет WMT14-данных и
   бюджета (8×P100, 3.5 дня) — не заявляем.
2. **Влияние отдельного обучения embeddings / sinusoidal PE / beam search**:
   не пробовали варианты — в рамках одного baseline-прогона.
3. **Чувствительность к seed**: один прогон (seed 42); разброс по сиду
   неизвестен.
4. **Сходимость к plateu**: 30 эпох без early stopping (best epoch 26,
   `metrics.json`); не исключено улучшение при большем бюджете — это
   отличает наш 22.11 от эталонного 13.09 (эталон: 10 эпох, best 5).

## Изменения кода в ходе отладки (4 исправления)

1. **CUDA вместо MPS**: `train.py:129–134` выбирает cuda → mps → cpu;
   README/заголовок ориентированы на MPS
   (`PYTORCH_ENABLE_MPS_FALLBACK=1`), но реальный прогон пошёл по CUDA —
   `logs/train.log` строка 1: `device=cuda torch=2.12.1+cu130`.
2. **Формула Noam LR**: `model.py:100–106` — `d_model^-0.5 *
   min(step^-0.5, step*warmup^-1.5)`, вызов `train.py:209–211` (lr
   устанавливается вручную в param_groups); сходимость с формулой (3)
   статьи p. 7 проверена по значениям lr в `logs/train.log`.
3. **BOS/EOS в target**: `train.py:34` — `PAD, BOS, EOS = 0, 2, 3`;
   target = `[BOS]+ids+[EOS]`, обучение на сдвиг `tgt[:, :-1] → tgt[:, 1:]`
   (`train.py:203`); greedy decode стартует с BOS и останавливается на EOS
   (`model.py:73–89`). Совпадает с эталоном.
4. **sacreBLEU: один референс-поток**: `train.py:94–96` —
   `sacrebleu.corpus_bleu(hyps, [refs])`, где `refs` — список
   референс-фраз, обёрнутый в один поток (сacreBLEU ожидает список
   референсов); токенизация 13a по умолчанию (README).

## Замечания по данным/артефактам

- `artifacts/` содержит только `bpe.json`; `best.pt` должен писаться
  (`train.py:256–257`), но в каталоге его нет — чекпоинт не сохранился
  (метрики в `metrics.json` уже посчитаны из этого чекпоинта в процессе
  прогона).
- Ошибка в `recipe.json` (исправлена в этой сессии): `reported_bleu.base_en_de`
  было 25.8 (p. 9 — это Table 3, newstest2013 dev); правильно **27.3, p. 8**
  (Table 2, newstest2014) — по выдаче MCP.
- train_loss 51.96 @ epoch 1 — не аномалия: embeddings N(0,1)·scale=16 дают
  большие logits на старте → −log_softmax очень отрицателен.
