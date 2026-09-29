"""Convert an eagle-train DeepSpeed checkpoint to the released EAGLE drafter format.

The output directory (pytorch_model.bin + config.json, same tensor names as mit-han-lab/Qwen2.5-7B-Eagle-RL:
embed_tokens, fc, layers.0.*) loads anywhere the released drafter does: SGLang speculative_draft_model_path,
the RL launcher's SPEC_MODEL_PATH, bench_acceptance.sh's EAGLE_PATH.

Usage: python export_drafter.py --ckpt <eagle_trainer output_dir> --out <dir> [--tag global_stepN]
"""

import argparse
import json
import os
import shutil

import torch
from huggingface_hub import snapshot_download


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default=None, help="checkpoint tag (default: the 'latest' file)")
    ap.add_argument("--reference", default="mit-han-lab/Qwen2.5-7B-Eagle-RL")
    args = ap.parse_args()

    tag = args.tag or open(os.path.join(args.ckpt, "latest")).read().strip()
    module = torch.load(os.path.join(args.ckpt, tag, "mp_rank_00_model_states.pt"), map_location="cpu",
                        weights_only=False)["module"]
    ref_dir = snapshot_download(args.reference)
    ref = torch.load(os.path.join(ref_dir, "pytorch_model.bin"), map_location="cpu", mmap=True, weights_only=True)

    out, report = {}, {}
    for k, v in ref.items():
        src = k if k in module else f"model.{k}"
        w = module[src]
        assert tuple(w.shape) == tuple(v.shape), (k, w.shape, v.shape)
        out[k] = w.to(v.dtype).contiguous()
        report[k] = float((out[k].float() - v.float()).abs().max())
    os.makedirs(args.out, exist_ok=True)
    torch.save(out, os.path.join(args.out, "pytorch_model.bin"))
    shutil.copy(os.path.join(ref_dir, "config.json"), os.path.join(args.out, "config.json"))
    json.dump({"source_checkpoint": os.path.join(args.ckpt, tag), "reference": ref_dir,
               "max_abs_change_vs_reference": report}, open(os.path.join(args.out, "export_info.json"), "w"), indent=2)
    print(f"exported {len(out)} tensors from {args.ckpt}/{tag} to {args.out}")
    for k, d in report.items():
        print(f"  {k}: max |w - w_ref| = {d:.3g}")
    assert report["embed_tokens.weight"] == 0.0, "embed_tokens must stay the frozen target copy"


if __name__ == "__main__":
    main()
