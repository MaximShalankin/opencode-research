"""AIAYN scaled reproduction (steps 3-4): Multi30k en->de, small Transformer.

Run from workspace/:
    PYTORCH_ENABLE_MPS_FALLBACK=1 python3 train.py
Smoke test (small subset, writes logs/smoke.log only, no metrics.json):
    python3 train.py --smoke

Outputs (full run):
    logs/train.log   per-epoch training log (loss, val BLEU, time)
    metrics.json     config + metrics; test_bleu from best val-BLEU checkpoint
    artifacts/bpe.json, artifacts/best.pt
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import json
import random
import resource
import time
from pathlib import Path

import numpy as np
import sacrebleu
import torch

import bpe as bpe_mod
from model import LabelSmoothingLoss, Transformer, noam_lr

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
LOGS = ROOT / "logs"
SEED = 42
PAD, BOS, EOS = 0, 2, 3
POS_MAX = 64
D_MODEL, NHEAD, LAYERS, D_FF = 256, 4, 2, 1024
DROPOUT, LABEL_SMOOTHING, WARMUP = 0.1, 0.1, 4000


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate(idx, srcs, tgts):
    s = [srcs[i] for i in idx]
    t = [tgts[i] for i in idx]
    Ls = max(len(x) for x in s)
    Lt = max(len(x) for x in t)
    sa = np.zeros((len(s), Ls), dtype=np.int64)
    ta = np.zeros((len(t), Lt), dtype=np.int64)
    for r, x in enumerate(s):
        sa[r, :len(x)] = x
    for r, x in enumerate(t):
        ta[r, :len(x)] = x
    src = torch.from_numpy(sa)
    tgt = torch.from_numpy(ta)
    return src, tgt, src == PAD


def dynamic_batches(order, src_lens, tgt_lens, batch_tokens, max_pairs=512):
    """Greedy packing of shuffled indices so src+tgt tokens <= batch_tokens."""
    batches, cur, total = [], [], 0
    for i in order:
        l = int(src_lens[i] + tgt_lens[i])
        if cur and total + l > batch_tokens:
            batches.append(cur)
            cur, total = [], 0
        cur.append(i)
        total += l
        if len(cur) >= max_pairs:
            batches.append(cur)
            cur, total = [], 0
    if cur:
        batches.append(cur)
    return [sorted(b, key=lambda i: -src_lens[i]) for b in batches]


def decode_split(model, tok, srcs, device, batch=256, max_gen=62):
    model.eval()
    hyps = []
    for i in range(0, len(srcs), batch):
        idx = list(range(i, min(i + batch, len(srcs))))
        src, _tgt, mask = collate(idx, srcs, srcs)
        src, mask = src.to(device), mask.to(device)
        ids = model.greedy_decode(src, mask, max_gen)
        hyps.extend(tok.decode(x).strip() for x in ids)
    return hyps


def bleu_score(hyps, refs):
    # sacrebleu ждёт список референс-потоков: один поток = все фразы.
    return sacrebleu.corpus_bleu(hyps, [refs]).score


def peak_memory_mb(device):
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1e6
    if device.type == "mps":
        try:
            return torch.mps.max_memory_allocated() / 1e6
        except Exception:
            pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main():
    ap = argparse.ArgumentParser(description="AIAYN scaled: Multi30k en->de")
    ap.add_argument("--smoke", action="store_true",
                    help="small subset (2000 train / 300 val, 4 epochs), "
                         "writes logs/smoke.log only")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--min-epochs", type=int, default=3)
    ap.add_argument("--batch-tokens", type=int, default=2048)
    args = ap.parse_args()

    t0 = time.time()
    lines = []

    def log(msg):
        print(msg, flush=True)
        lines.append(msg)

    seed_everything()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    log(f"config: seed={SEED} device={device} torch={torch.__version__} "
        f"layers={LAYERS} d_model={D_MODEL} heads={NHEAD} d_ff={D_FF} "
        f"dropout={DROPOUT} label_smoothing={LABEL_SMOOTHING} "
        f"warmup_steps={WARMUP} batch_tokens={args.batch_tokens} "
        f"epochs={args.epochs} patience={args.patience} min_epochs={args.min_epochs}")

    tok = bpe_mod.load_tokenizer()
    vocab = tok.get_vocab_size()
    log(f"tokenizer: vocab={vocab} (artifacts/bpe.json)")

    data = {}
    refs = {}
    for split in ("train", "val", "test"):
        src, tgt, raw, dropped = bpe_mod.tokenize_split(split, tok, POS_MAX)
        data[split] = (src, tgt)
        refs[split] = raw
        mx = max([max(map(len, src), default=0), max(map(len, tgt), default=0)])
        log(f"data {split}: pairs={len(src)} max_tokens={mx} dropped_long={dropped}")

    src_tr, tgt_tr = data["train"]
    src_va, tgt_va = data["val"]
    src_te, tgt_te = data["test"]

    if args.smoke:
        perm = np.random.permutation(len(src_tr))[:2000]
        src_tr = [src_tr[i] for i in perm]
        tgt_tr = [tgt_tr[i] for i in perm]
        src_va, refs_va = src_va[:300], refs["val"][:300]
        args.epochs = min(args.epochs, 4)
        log(f"smoke: train={len(src_tr)} val={len(src_va)} epochs={args.epochs}")
    else:
        refs_va, refs_te = refs["val"], refs["test"]

    src_lens = np.array([len(x) for x in src_tr])
    tgt_lens = np.array([len(x) for x in tgt_tr])
    train_tokens_per_epoch = int((src_lens + tgt_lens).sum())
    log(f"train: {train_tokens_per_epoch} src+tgt tokens/epoch, "
        f"~{max(1, train_tokens_per_epoch // args.batch_tokens)} steps/epoch")

    model = Transformer(vocab, d_model=D_MODEL, nhead=NHEAD, num_layers=LAYERS,
                        d_ff=D_FF, dropout=DROPOUT, max_len=POS_MAX,
                        pad_id=PAD, bos_id=BOS, eos_id=EOS).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"model: params={n_params / 1e6:.2f}M")

    loss_fn = LabelSmoothingLoss(vocab, smoothing=LABEL_SMOOTHING,
                                 ignore_index=PAD).to(device)
    opt = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9)

    step = 0
    total_tokens = 0
    val_bleus = []
    train_losses = []
    best = {"bleu": -1.0, "epoch": -1}
    best_state = None
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        e0 = time.time()
        model.train()
        order = np.random.permutation(len(src_tr))
        batches = dynamic_batches(order, src_lens, tgt_lens, args.batch_tokens)
        loss_sum, tok_sum = 0.0, 0
        for b in batches:
            src, tgt, mask = collate(b, src_tr, tgt_tr)
            src = src.to(device)
            tgt = tgt.to(device)
            mask = mask.to(device)
            tgt_in, labels = tgt[:, :-1], tgt[:, 1:]
            logits = model(src, mask, tgt_in, tgt_in == PAD)
            loss = loss_fn(logits, labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            step += 1
            lr = noam_lr(step, D_MODEL, WARMUP)
            for g in opt.param_groups:
                g["lr"] = lr
            opt.step()
            valid = int((labels != PAD).sum())
            loss_sum += float(loss) * valid
            tok_sum += valid
            total_tokens += int(src.numel() + tgt.numel())
        avg_loss = loss_sum / max(tok_sum, 1)
        train_losses.append(avg_loss)
        val_hyps = decode_split(model, tok, src_va, device)
        vb = bleu_score(val_hyps, refs_va)
        val_bleus.append(vb)
        log(f"epoch {epoch}/{args.epochs} | train_loss {avg_loss:.4f} | "
            f"val_bleu {vb:.2f} | steps={step} | lr={lr:.6f} | "
            f"t={time.time() - e0:.1f}s")
        if vb > best["bleu"]:
            best = {"bleu": vb, "epoch": epoch}
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
        if epoch >= args.min_epochs and epoch - best["epoch"] >= args.patience:
            log(f"early stopping: best epoch {best['epoch']} "
                f"(val_bleu {best['bleu']:.2f}), no improvement for "
                f"{args.patience} epochs")
            break

    if args.smoke:
        log(f"smoke done in {time.time() - t0:.0f}s, "
            f"last val_bleu={val_bleus[-1]:.2f}")
        LOGS.mkdir(parents=True, exist_ok=True)
        (LOGS / "smoke.log").write_text("\n".join(lines) + "\n")
        return

    model.load_state_dict(best_state)
    best_state = None
    test_hyps = decode_split(model, tok, src_te, device)
    test_bleu = bleu_score(test_hyps, refs_te)
    wall = time.time() - t0
    train_wall = time.time() - start

    log(f"best epoch {best['epoch']} (val_bleu {best['bleu']:.2f})")
    log(f"test_bleu (sacrebleu 13a, test_2016_flickr en->de) = {test_bleu:.2f}")
    log(f"wall_clock={wall:.0f}s train={train_wall:.0f}s "
        f"train_tokens={total_tokens} "
        f"tokens_per_sec={total_tokens / wall:.0f}")

    ART.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()},
               ART / "best.pt")

    metrics = {
        "seed": SEED,
        "config": {
            "layers": LAYERS,
            "d_model": D_MODEL,
            "heads": NHEAD,
            "d_ff": D_FF,
            "dropout": DROPOUT,
            "label_smoothing": LABEL_SMOOTHING,
            "warmup_steps": WARMUP,
            "vocab_size": vocab,
            "batch_tokens": args.batch_tokens,
        },
        "epochs": len(val_bleus),
        "train_loss": round(train_losses[best["epoch"] - 1], 4),
        "val_bleu": [round(x, 2) for x in val_bleus],
        "test_bleu": round(test_bleu, 2),
        "wall_clock_sec": round(wall, 1),
        "tokens_per_sec": round(total_tokens / wall, 1),
        "peak_memory_mb": round(peak_memory_mb(device), 1),
    }
    (ROOT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / "train.log").write_text("\n".join(lines) + "\n")
    log("metrics.json and logs/train.log written")


if __name__ == "__main__":
    main()
