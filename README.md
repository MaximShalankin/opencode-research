# opencode-research

Демо исследовательского агента на [OpenCode](https://opencode.ai): агент
изучает научную статью через долговременную память, извлекает рецепт
эксперимента, пишет код, обучает модель, считает метрику и оформляет отчёт
с доказательствами.

Идея — пройти полный путь **«статья → baseline-метрики»** силами агента,
без ручного переписывания метода из публикации.

## Как это работает

1. **Ingest** — PDF статьи конвертируется в markdown с якорями страниц.
2. **Memory** — [vault-core](https://github.com/MaximShalankin/vault-agents)
   индексирует статью; агент ищет по ней через MCP и цитирует страницы.
3. **Agent** — OpenCode получает задачу (`task.md`) и работает в `workspace/`.
4. **Artifacts** — `recipe.json`, `PLAN.md`, код, `metrics.json`, `REPORT.md`.

Подробнее: [docs/how-it-works.md](docs/how-it-works.md).

## Быстрый старт

```bash
# 1. Память (vault-core)
git clone https://github.com/MaximShalankin/vault-agents ~/vault-agents
(cd ~/vault-agents && uv sync)
export PATH="$HOME/vault-agents/.venv/bin:$PATH"

# 2. Статья → markdown → индекс (из корня репозитория)
curl -L -o paper.pdf https://arxiv.org/pdf/1706.03762
python3 tools/paper_ingest.py paper.pdf --out papers/aiayn --slug aiayn
vault-core init      # скелет/MCP; конфиг памяти уже в репозитории
vault-core reindex

# 3. Прогон агента
cp opencode.json.example opencode.json   # укажите baseURL и id модели
opencode run "$(cat task.md)"
```

Требования: Python 3.12, poppler (`pdftotext`), OpenCode 1.19+,
OpenAI-совместимый сервер с Qwen3.8-27B и tool-calling.
Память и запуск — [docs/memory.md](docs/memory.md).

## Пример

[`examples/aiayn/`](examples/aiayn/README.md) — воспроизведение
«Attention Is All You Need» в уменьшенном масштабе: Multi30k, малый
Transformer, test BLEU 22.11 и отчёт с цитатами страниц и разделом
«что не воспроизведено».

## Лицензия

MIT — см. [LICENSE](LICENSE).
