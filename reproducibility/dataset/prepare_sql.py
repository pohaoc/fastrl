"""Prepare SkyRL-SQL-653 for FastRL, as used by the granular-cais-rl harness.

Downloads NovaSky-AI/SkyRL-SQL-653-data-newfmt (prompts, gold SQL) and the OmniSQL database bundle
(seeklhy/OmniSQL-datasets data.zip, 22 GB), extracts only the SQLite databases the data references,
and writes verl-format parquets. Prompts are kept exactly as published (system template + user
message). The SkyRL-gym ``text2sql`` env extras travel in ``extra_info.tools_kwargs``.

Usage: python reproducibility/dataset/prepare_sql.py [--out SkyRL-SQL] [--raw SkyRL-SQL-raw]   (run from the repo root)
Then point ``text2sql.db_path`` at ``<out>/db/data``.
"""

import argparse
import os
import zipfile

import pandas as pd
from huggingface_hub import hf_hub_download, snapshot_download


def to_verl(df: pd.DataFrame, split: str) -> pd.DataFrame:
    rows = []
    for i, r in enumerate(df.itertuples(index=False)):
        rows.append(
            {
                "data_source": "skyrl_sql",
                "prompt": [{"role": m["role"], "content": m["content"]} for m in r.prompt],
                "ability": "text2sql",
                "reward_model": {"style": "rule", "ground_truth": r.reward_spec["ground_truth"]},
                "extra_info": {
                    "index": i,
                    "split": split,
                    "tools_kwargs": {
                        "env_class": r.env_class,
                        "db_id": r.db_id,
                        "data": r.data,
                        "reward_spec": {
                            "method": r.reward_spec["method"],
                            "ground_truth": r.reward_spec["ground_truth"],
                        },
                    },
                },
            }
        )
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="SkyRL-SQL")
    ap.add_argument("--raw", default="SkyRL-SQL-raw")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    snapshot_download("NovaSky-AI/SkyRL-SQL-653-data-newfmt", repo_type="dataset", local_dir=args.raw)
    splits = {s: pd.read_parquet(os.path.join(args.raw, f"{s}.parquet")) for s in ("train", "validation")}
    for s, df in splits.items():
        to_verl(df, s).to_parquet(os.path.join(args.out, f"{s}.parquet"))
        print(f"{s}: {len(df)} rows, sources {df['data'].value_counts().to_dict()}")

    # Extract only the databases the data references (the full bundle is 22 GB compressed).
    zip_path = hf_hub_download(
        "seeklhy/OmniSQL-datasets", "data.zip", repo_type="dataset", local_dir=os.path.join(args.raw, "omnisql")
    )
    needed = {(r.data, r.db_id) for df in splits.values() for r in df.itertuples(index=False)}
    prefix = {"synsql": "data/SynSQL-2.5M/databases/", "spider": "data/spider/database/"}
    wanted_dirs = {prefix[d] + db + "/" for d, db in needed}
    with zipfile.ZipFile(zip_path) as z:
        members = [n for n in z.namelist() if any(n.startswith(w) for w in wanted_dirs)]
        z.extractall(os.path.join(args.out, "db"), members=members)
    root = os.path.join(args.out, "db", "data")
    sub = {"synsql": "SynSQL-2.5M/databases", "spider": "spider/database"}
    missing = [db for d, db in needed if not os.path.exists(os.path.join(root, sub[d], db, db + ".sqlite"))]
    if missing:
        raise SystemExit(f"missing databases: {missing}")
    print(f"extracted {len(needed)} databases to {root}")


if __name__ == "__main__":
    main()
