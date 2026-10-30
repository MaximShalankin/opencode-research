"""Small Transformer (AIAYN-style) for en->de, with label-smoothing loss and
the Noam warmup/inverse-sqrt LR schedule. Tied input/output embeddings,
learned positional embeddings.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size, smoothing=0.1, ignore_index=0):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        log_probs = F.log_softmax(logits, dim=-1)
        nll = -log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        smooth = -log_probs.mean(dim=-1)
        loss = self.confidence * nll + self.smoothing * smooth
        pad = target == self.ignore_index
        loss = loss.masked_fill(pad, 0.0)
        denom = (target.numel() - pad.sum()).clamp(min=1)
        return loss.sum() / denom


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=2, d_ff=1024,
                 dropout=0.1, max_len=64, pad_id=0, bos_id=2, eos_id=3):
        super().__init__()
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.scale = d_model ** 0.5
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.src_pos = nn.Embedding(max_len, d_model)
        self.tgt_pos = nn.Embedding(max_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, d_ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        dec_layer = nn.TransformerDecoderLayer(
            d_model, nhead, d_ff, dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers)
        self.out_proj = nn.Linear(d_model, vocab_size, bias=False)
        self.out_proj.weight = self.embed.weight
        nn.init.normal_(self.embed.weight)
        nn.init.normal_(self.src_pos.weight)
        nn.init.normal_(self.tgt_pos.weight)

    def _encoder(self, src, src_pad_mask):
        T = src.size(1)
        x = self.embed(src) * self.scale
        x = x + self.src_pos(torch.arange(T, device=src.device))
        return self.encoder(x, src_key_padding_mask=src_pad_mask)

    def forward(self, src, src_pad_mask, tgt_in, tgt_pad_mask):
        mem = self._encoder(src, src_pad_mask)
        T = tgt_in.size(1)
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool,
                                       device=src.device), diagonal=1)
        x = self.embed(tgt_in) * self.scale
        x = x + self.tgt_pos(torch.arange(T, device=src.device))
        out = self.decoder(x, mem, tgt_mask=causal,
                           tgt_key_padding_mask=tgt_pad_mask)
        return self.out_proj(out)

    @torch.no_grad()
    def greedy_decode(self, src, src_pad_mask, max_len):
        """Greedy decoding; returns list of token-id lists (BOS/EOS stripped)."""
        self.eval()
        mem = self._encoder(src, src_pad_mask)
        B = src.size(0)
        cur = torch.full((B, 1), self.bos_id, dtype=torch.long,
                         device=src.device)
        finished = torch.zeros(B, dtype=torch.bool, device=src.device)
        out = torch.zeros(B, max_len, dtype=torch.long, device=src.device)
        for step in range(max_len):
            T = cur.size(1)
            causal = torch.triu(torch.ones(T, T, dtype=torch.bool,
                                           device=src.device), diagonal=1)
            x = self.embed(cur) * self.scale
            x = x + self.tgt_pos(torch.arange(T, device=src.device))
            logits = self.out_proj(self.decoder(x, mem, tgt_mask=causal))[:, -1]
            nxt = logits.argmax(dim=-1)
            nxt = torch.where(finished, torch.full_like(nxt, self.eos_id), nxt)
            cur = torch.cat([cur, nxt.unsqueeze(1)], dim=1)
            out[:, step] = nxt
            finished |= nxt == self.eos_id
            if finished.all():
                break
        hyps = []
        for i in range(B):
            row = out[i]
            e = (row == self.eos_id).nonzero()
            end = int(e[0]) if len(e) else max_len
            hyps.append(row[:end].tolist())
        return hyps


def noam_lr(step, d_model, warmup):
    """Paper schedule: linear warmup then inverse square-root decay.

    lrate = d_model^-0.5 * min(step^-0.5, step * warmup^-1.5)
    """
    step = max(1, step)
    return (d_model ** -0.5) * min(step ** -0.5, step * warmup ** -1.5)
