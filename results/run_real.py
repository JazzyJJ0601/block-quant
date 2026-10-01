#!/usr/bin/env python3
"""Real results: 4-bit weight quantisation of every decoder linear layer in Qwen3-8B.

Methods (all weight-only, symmetric, levels -7..7):
  fp16          no quantisation
  rtn_channel   one absmax scale per output row (the standard baseline)
  block_absmax  one absmax scale per 64 weights
  block_mse     per-64-block scale chosen by clip search, minimising weight error
  block_act     per-64-block clip search weighted by mean squared input activation
                (calibrated on WikiText-2 *train*), so blocks protect the weights
                that move the layer's output most

Perplexity on WikiText-2 *test*, N_WINDOWS non-overlapping windows of SEQ tokens.
Each method's result is appended to results/real.json as soon as it finishes.
"""
import gc
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from block_quant import quantize_with_scales, search_scales  # noqa: E402

MODEL_PATH = "/home/jasper/eirene-projects/03-inference-lab/ai-lab/models/Qwen--Qwen3-8B"
BITS, BLOCK = 4, 64
SEQ, N_WINDOWS = 512, 40
N_CALIB = 8
OUT = HERE / "real.json"


def load_text(split):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    return "\n\n".join(ds["text"])


def windows(tok, text, n):
    ids = tok(text, return_tensors="pt").input_ids[0]
    return [ids[i * SEQ:(i + 1) * SEQ] for i in range(n)]


@torch.no_grad()
def perplexity(model, wins):
    nll, count = 0.0, 0
    for w in wins:
        x = w.unsqueeze(0).cuda()
        loss = model(x, labels=x).loss.item()
        nll += loss * (x.numel() - 1)
        count += x.numel() - 1
    return float(torch.exp(torch.tensor(nll / count)))


def decoder_linears(model):
    return [(n, m) for n, m in model.named_modules()
            if isinstance(m, nn.Linear) and ".layers." in n]


@torch.no_grad()
def act_stats(model, linears, calib):
    """Mean squared input activation per input channel, for each linear."""
    sums, hooks = {}, []
    for name, mod in linears:
        def hook(m, inp, out, name=name):
            a = inp[0].float().pow(2).reshape(-1, inp[0].shape[-1]).mean(0)
            sums[name] = sums.get(name, 0) + a
        hooks.append(mod.register_forward_hook(hook))
    for w in calib:
        model(w.unsqueeze(0).cuda())
    for h in hooks:
        h.remove()
    return {k: v / len(calib) for k, v in sums.items()}


@torch.no_grad()
def quantise(w, method, act=None):
    w32 = w.float()
    qmax = 2 ** (BITS - 1) - 1
    if method == "rtn_channel":
        s = (w32.abs().amax(dim=1, keepdim=True) / qmax).clamp_min(1e-12)
        return (torch.clamp(torch.round(w32 / s), -qmax, qmax) * s).to(w.dtype)
    if method == "block_absmax":
        s = w32.reshape(-1, BLOCK).abs().amax(dim=1) / qmax
    elif method == "block_mse":
        s = search_scales(w32, BITS, BLOCK)
    elif method == "block_act":
        s = search_scales(w32, BITS, BLOCK, weight=act.view(1, -1))
    return quantize_with_scales(w32, s, BITS, BLOCK).to(w.dtype)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, local_files_only=True, device_map="cuda")
    model.eval()
    test = windows(tok, load_text("test"), N_WINDOWS)
    calib = windows(tok, load_text("train"), N_CALIB)
    linears = decoder_linears(model)
    print(f"{len(linears)} decoder linear layers", flush=True)

    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    results["setup"] = {"model": "Qwen3-8B", "bits": BITS, "block": BLOCK, "layers": len(linears),
                        "eval": f"WikiText-2 test, {N_WINDOWS}x{SEQ} tokens", "calib": f"WikiText-2 train, {N_CALIB}x{SEQ} tokens"}
    originals = {n: m.weight.data.to("cpu", copy=True) for n, m in linears}
    acts = act_stats(model, linears, calib)

    for method in ["fp16", "rtn_channel", "block_absmax", "block_mse", "block_act"]:
        if method in results:
            print(method, "already done", results[method], flush=True)
            continue
        t0 = time.time()
        for n, m in linears:
            m.weight.data = originals[n].cuda()
            if method != "fp16":
                m.weight.data = quantise(m.weight.data, method, acts.get(n))
        ppl = perplexity(model, test)
        results[method] = {"ppl": round(ppl, 4), "seconds": round(time.time() - t0, 1)}
        OUT.write_text(json.dumps(results, indent=2))
        print(method, results[method], flush=True)
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
