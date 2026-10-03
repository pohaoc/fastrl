"""Convert an eagle-train DeepSpeed checkpoint to the released EAGLE drafter format.

The output directory (pytorch_model.bin + config.json) loads anywhere the released drafter does: SGLang
speculative_draft_model_path, SPEC_MODEL_PATH in the RL launcher.
  - tensors: same names as --layout (mit-han-lab/Qwen2.5-7B-Eagle-RL: embed_tokens, fc, layers.0.*);
  - config.json: the TARGET's config with architectures=[Qwen2ForCausalLMEagle] and num_hidden_layers=1, which
    is exactly how the released drafter's config relates to Qwen2.5-7B's.
embed_tokens must equal the target's (eagle-train loads it from the target and freezes it); this is checked.

Usage: python export_drafter.py --ckpt <eagle_trainer output_dir> --out <dir> --target <target snapshot dir>
           --layout <released drafter dir> [--init <warm-start drafter dir>] [--tag global_stepN]
"""

import argparse
import json
import os

import torch
from safetensors import safe_open


def target_embed(target_dir):
    index = json.load(open(os.path.join(target_dir, "model.safetensors.index.json")))
    f = index["weight_map"]["model.embed_tokens.weight"]
    with safe_open(os.path.join(target_dir, f), framework="pt", device="cpu") as h:
        return h.get_tensor("model.embed_tokens.weight")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", required=True, help="target model snapshot dir")
    ap.add_argument("--layout", required=True, help="released EAGLE drafter dir whose tensor names to reproduce")
    ap.add_argument("--init", default=None, help="warm-start drafter dir, to report how far training moved")
    ap.add_argument("--tag", default=None, help="checkpoint tag (default: the 'latest' file)")
    args = ap.parse_args()

    tag = args.tag or open(os.path.join(args.ckpt, "latest")).read().strip()
    module = torch.load(os.path.join(args.ckpt, tag, "mp_rank_00_model_states.pt"), map_location="cpu",
                        weights_only=False)["module"]
    layout = torch.load(os.path.join(args.layout, "pytorch_model.bin"), map_location="cpu", mmap=True, weights_only=True)
    init = (torch.load(os.path.join(args.init, "pytorch_model.bin"), map_location="cpu", mmap=True, weights_only=True)
            if args.init else {})

    out, report = {}, {}
    for k, v in layout.items():
        src = k if k in module else f"model.{k}"
        w = module[src]
        assert tuple(w.shape) == tuple(v.shape), (k, w.shape, v.shape)
        out[k] = w.to(v.dtype).contiguous()
        if k in init and "embed_tokens" not in k:
            report[k] = float((out[k].float() - init[k].float()).abs().max())
    embed_diff = float((out["embed_tokens.weight"].float() - target_embed(args.target).float()).abs().max())
    assert embed_diff == 0.0, f"embed_tokens differs from the target's by {embed_diff}: not trained against this target"

    cfg = json.load(open(os.path.join(args.target, "config.json")))
    cfg["architectures"] = ["Qwen2ForCausalLMEagle"]
    cfg["num_hidden_layers"] = 1
    os.makedirs(args.out, exist_ok=True)
    torch.save(out, os.path.join(args.out, "pytorch_model.bin"))
    json.dump(cfg, open(os.path.join(args.out, "config.json"), "w"), indent=2)
    json.dump({"source_checkpoint": os.path.join(args.ckpt, tag), "target": args.target, "layout": args.layout,
               "init": args.init, "max_abs_change_vs_init": report}, open(os.path.join(args.out, "export_info.json"), "w"),
              indent=2)
    print(f"exported {len(out)} tensors from {args.ckpt}/{tag} to {args.out}; embed_tokens == target's")
    for k, d in report.items():
        print(f"  {k}: max |w - w_init| = {d:.3g}")


if __name__ == "__main__":
    main()
