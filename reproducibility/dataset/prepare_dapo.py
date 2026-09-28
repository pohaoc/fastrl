"""Prepare DAPO-Math-17k (train) and AIME-2024 (validation) for FastRL.

The published parquets repeat every problem many times (17,398 unique prompts in 1.79M rows;
AIME-2024 is 30 problems x 32), so keep the first copy of each prompt.

Usage: python reproducibility/dataset/prepare_dapo.py [--out DAPO-Math-17k]   (run from the repo root)
"""

import argparse
import os

import pandas as pd
from huggingface_hub import hf_hub_download

SOURCES = {
    "train": ("BytedTsinghua-SIA/DAPO-Math-17k", "data/dapo-math-17k.parquet"),
    "validation": ("BytedTsinghua-SIA/AIME-2024", "data/aime-2024.parquet"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="DAPO-Math-17k")
    ap.add_argument("--raw", default="DAPO-Math-17k-raw")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for split, (repo, name) in SOURCES.items():
        path = hf_hub_download(repo, name, repo_type="dataset", local_dir=args.raw)
        df = pd.read_parquet(path)
        key = df["prompt"].map(lambda msgs: "\n".join(m["content"] for m in msgs))
        dedup = df.loc[~key.duplicated()].reset_index(drop=True)
        dedup.to_parquet(os.path.join(args.out, f"{split}.parquet"))
        print(f"{split}: {len(df)} rows -> {len(dedup)} unique prompts ({repo})")


if __name__ == "__main__":
    main()
