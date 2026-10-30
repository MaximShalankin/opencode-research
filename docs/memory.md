# Память: vault-core + MCP

Агент ищет по статье через [vault-core](https://github.com/MaximShalankin/vault-agents) —
локальный RAG-сервер с MCP-интерфейсом.

## Установка

```bash
git clone https://github.com/MaximShalankin/vault-agents ~/vault-agents
cd ~/vault-agents
uv sync
export PATH="$HOME/vault-agents/.venv/bin:$PATH"
```

`vault-core` должен быть доступен в `PATH` процессу OpenCode. Если это
неудобно, укажите в `opencode.json` абсолютный путь к бинарнику вместо
`vault-core`.

## Проект памяти

Из корня этого репозитория:

```bash
vault-core init
```

Команда создаёт `vault/00_System/`, конфиг и `.mcp.json` (идемпотентно).

## Статья

```bash
curl -L -o paper.pdf https://arxiv.org/pdf/1706.03762
python3 tools/paper_ingest.py paper.pdf --out papers/aiayn --slug aiayn
```

Получится `papers/aiayn/aiayn.md` с якорями `<!-- page: N -->` и
`meta.json` (sha256 PDF и markdown).

## Конфиг

Конфиг уже лежит в репозитории — `vault/00_System/vault.config.yaml`:

```yaml
include_dirs:
  - papers
chunk_boundary: page
```

`chunk_boundary: page` — чанк не пересекает границу страницы, номер
страницы попадает в выдачу (`p. N`). Это и делает цитаты проверяемыми.
Менять ничего не нужно; `vault-core init` существующий конфиг не трогает.

## Индекс

```bash
vault-core reindex
vault-core status
```

Эмбеддинги — BGE-M3 через Ollama (`ollama pull bge-m3`). Без Ollama
поиск идёт в BM25-режиме (в конфиге можно явно поставить
`retrieval.mode: bm25`).

**Важно:** Chroma не безопасна между процессами. Полный `reindex` —
только при остановленном MCP; при живом MCP используйте инкрементальный
`reindex` через сам MCP.

## MCP в OpenCode

```json
{
  "mcp": {
    "vault": {
      "type": "local",
      "command": ["vault-core", "mcp", "--project", "."],
      "enabled": true
    }
  }
}
```

Агенту доступны инструменты `vault_search`, `vault_get_note`,
`vault_status`, `vault_reindex` (и `vault_kg_*`). Для статьи нужны
первые два; в выдаче `vault_search` есть `source=papers` и `p. N`.

## Данные для примера

Датасет Multi30k (task1, en→de) не входит в репозиторий. Скачайте
архивы из [multi30k/dataset](https://github.com/multi30k/dataset/tree/master/data/task1/raw)
и распакуйте в `data/multi30k/`:

```bash
mkdir -p data/multi30k && cd data/multi30k
base=https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw
for f in train.en train.de val.en val.de test_2016_flickr.en test_2016_flickr.de; do
  curl -LO "$base/$f.gz" && gunzip "$f.gz"
done
```

Проверенные sha256 архивов:

```
d79cfa999dd4c51d2cb42499b6796d5a882c3a8a961923c25a898c90f8bbd56f  train.en.gz
726e39b2fa9eb9ffb6dc763fb35a179f80fae06ffc5d28b6ace5faa883de28a6  train.de.gz
14f7d25ddd868909a9213e361768460edcacdd6ab9d1e77b92560dc10c10dc28  val.en.gz
f0cba2f995189cf5770f29a8a9a537a3ad3f51657ad873405082ff6863a5e75a  val.de.gz
611d361c6334bc7246101d48097c13cf5c4413c5befc793cc629934359d532d9  test_2016_flickr.en.gz
9204244e408ccb38d2a55cfcd344df15005fc42a07a6e55ca6c52b6ababb8cc8  test_2016_flickr.de.gz
```
