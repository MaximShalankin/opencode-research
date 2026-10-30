"""BPE tokenizer (shared en-de vocab, 16k) + tokenization of Multi30k splits.

Usage: imported by train.py; builds artifacts/bpe.json on first run.
"""
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers, trainers

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data" / "multi30k"
ART = ROOT / "artifacts"

SPECIAL = ["<pad>", "<unk>", "<s>", "</s>"]
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3
VOCAB_SIZE = 16000

SPLIT_FILES = {
    "train": "train",
    "val": "val",
    "test": "test_2016_flickr",
}


def build_tokenizer() -> Tokenizer:
    docs = []
    for name in ("train.en", "train.de"):
        docs += (DATA / name).read_text().splitlines()
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=SPECIAL,
    )
    tok.train_from_iterator(docs, trainer=trainer)
    ART.mkdir(parents=True, exist_ok=True)
    tok.save(str(ART / "bpe.json"))
    return tok


def load_tokenizer() -> Tokenizer:
    p = ART / "bpe.json"
    if not p.exists():
        return build_tokenizer()
    return Tokenizer.from_file(str(p))


def read_pairs(name: str):
    en = (DATA / f"{name}.en").read_text().splitlines()
    de = (DATA / f"{name}.de").read_text().splitlines()
    if len(en) != len(de):
        raise ValueError(f"line count mismatch for {name}: {len(en)} vs {len(de)}")
    return [(a, b) for a, b in zip(en, de) if a.strip() and b.strip()]


def tokenize_split(split: str, tok: Tokenizer, max_len: int):
    """Return (src_ids, tgt_ids, raw_tgt_text, n_dropped).

    tgt_ids include BOS/EOS: [<s>, t1, ..., tn, </s>].
    Pairs longer than max_len (in tokens) are dropped and counted.
    """
    name = SPLIT_FILES[split]
    pairs = read_pairs(name)
    src_enc = tok.encode_batch([p[0] for p in pairs], add_special_tokens=False)
    tgt_enc = tok.encode_batch([p[1] for p in pairs], add_special_tokens=False)
    src = [e.ids for e in src_enc]
    # BPE-постпроцессор не задан: BOS/EOS добавляем явно (train.py и
    # greedy_decode рассчитывают на них).
    tgt = [[BOS_ID] + e.ids + [EOS_ID] for e in tgt_enc]
    kept_s, kept_t, raw, dropped = [], [], [], 0
    for s, t, txt in zip(src, tgt, (p[1] for p in pairs)):
        if len(s) > max_len or len(t) > max_len:
            dropped += 1
            continue
        kept_s.append(s)
        kept_t.append(t)
        raw.append(txt)
    return kept_s, kept_t, raw, dropped
