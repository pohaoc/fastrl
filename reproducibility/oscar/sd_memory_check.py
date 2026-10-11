"""Check that the SD engine releases its drafter and CUDA graphs during training and restores them correctly.

One engine per process, built like verl's (enable_memory_saver, TP 4, mem 0.4, fa3, FastRL adaptive EAGLE args),
plus enable_weights_cpu_backup so the target weights come back without verl's weight push. Steps:
  1. greedy reference: --n_greedy prompts one at a time (token ids + speculative verify counts)
  2. GPU memory in use per GPU: active, after release(tags=["kv_cache","weights"]) as verl does, after resume
  3. --cycles x [release, resume, a sampled batch of --n_batch prompts that drains through every batch size]
  4. the greedy prompts again: outputs and verify counts must match step 1 exactly
FASTRL_RELEASE_SD_MEMORY=0 gives the old behaviour (drafter and graphs stay resident).

--rl_wakeup follows verl's RL path instead: the engine is released (all tags) right after start-up, and every
wake-up resumes "weights", pushes the target weights (here from the checkpoint, with update_weights_from_tensor),
then resumes "kv_cache". --push_target 0 keeps that order but restores the target from the CPU backup.
In that mode step 1's greedy pass runs after the first wake-up (compare its acceptance with a plain run's).
"""

import argparse
import json
import os
import subprocess
import time


SMI_INDEX = None  # set by --smi_index: report only the GPU this engine runs on


def gpu_used_mib():
    cmd = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    if SMI_INDEX is not None:
        cmd += ["-i", str(SMI_INDEX)]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return [int(x) for x in out.split()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--drafter", default=None)
    ap.add_argument("--sd", type=int, default=1)
    ap.add_argument("--n_greedy", type=int, default=6)
    ap.add_argument("--n_batch", type=int, default=64)
    ap.add_argument("--cycles", type=int, default=2)
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--mem", type=float, default=0.4, help="mem_fraction_static")
    ap.add_argument("--chunked_prefill", type=int, default=None)
    ap.add_argument("--rl_wakeup", type=int, default=0, help="verl's release / resume / push order (see above)")
    ap.add_argument("--push_target", type=int, default=1, help="with --rl_wakeup: push target weights on each wake-up")
    ap.add_argument("--smi_index", type=int, default=None, help="nvidia-smi index of the GPU this engine uses")
    ap.add_argument("--parquet", default=os.path.join(os.environ.get("DATA_ROOT", "."), "SkyRL-SQL", "train.parquet"))
    args = ap.parse_args()
    global SMI_INDEX
    SMI_INDEX = args.smi_index
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    import pandas as pd
    import sglang as sgl
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    df = pd.read_parquet(args.parquet).sample(args.n_greedy + args.n_batch, random_state=1)
    prompts = [tok.apply_chat_template(list(m), add_generation_prompt=True, tokenize=False) for m in df.prompt]
    greedy_p, batch_p = prompts[: args.n_greedy], prompts[args.n_greedy:]

    kw = dict(model_path=args.model, dtype="bfloat16", mem_fraction_static=args.mem, enable_memory_saver=True,
              enable_weights_cpu_backup=not (args.rl_wakeup and args.push_target), tp_size=args.tp, load_format="auto", mm_attention_backend="fa3",
              attention_backend="fa3", random_seed=0, log_level="warning")
    if args.sd:
        kw.update(speculative_algorithm="EAGLE", speculative_draft_model_path=args.drafter,
                  speculative_num_steps=8, speculative_eagle_topk=4, speculative_num_draft_tokens=48,
                  speculative_eagle_mab_algorithm="BEG", apdative_speculative_batch_size_threshold=32,
                  speculative_eagle_mab_configs=["8_4_32", "8_4_16", "8_4_8"], speculative_mab_bs_threshold=[1, 2, 5, 21],
                  max_running_requests=512, cuda_graph_max_bs=32, disable_overlap_schedule=True)
    if args.chunked_prefill:
        kw["chunked_prefill_size"] = args.chunked_prefill
    res = {"rl_wakeup": args.rl_wakeup, "push_target": args.push_target, "sd": args.sd, "tp": args.tp, "mem": args.mem, "release_sd_memory": os.environ.get("FASTRL_RELEASE_SD_MEMORY", "1"), "phase": "init"}
    save = lambda: json.dump(res, open(args.out, "w"), indent=2)  # noqa: E731
    save()
    mib0 = gpu_used_mib()
    engine = sgl.Engine(**kw)
    # engine.scheduler_info, not get_server_info(): the latter waits on a reply loop that SGLang only starts with the
    # first generate call, so right after start-up it blocks forever (job 7225251).
    info = engine.scheduler_info or {}
    res["engine"] = {k: info.get(k) for k in ("max_total_num_tokens", "max_req_input_len")}

    def push_target():  # verl pushes every target tensor after resuming "weights"
        import glob
        import torch
        from safetensors.torch import load_file
        for shard in sorted(glob.glob(os.path.join(args.model, "*.safetensors"))):
            sd = load_file(shard, device="cuda")
            engine.update_weights_from_tensor(list(sd.items()), flush_cache=False)
            del sd
            torch.cuda.empty_cache()
        engine.flush_cache()

    def wake_up():
        engine.resume_memory_occupation(tags=["weights"])
        if args.rl_wakeup and args.push_target:
            push_target()
        engine.resume_memory_occupation(tags=["kv_cache"])

    if args.rl_wakeup:  # verl: AsyncEngine releases everything before the first wake-up
        engine.release_memory_occupation()
        wake_up()

    def greedy():
        out = []
        for p in greedy_p:
            o = engine.generate(p, {"temperature": 0.0, "max_new_tokens": 512})
            out.append({"ids": list(o["output_ids"]), "verify_ct": o["meta_info"].get("spec_verify_ct", 0),
                        "tokens": o["meta_info"]["completion_tokens"]})
        return out

    res["phase"] = "greedy_before"; save()
    before = greedy()
    res["mem_mib"] = {"before_engine": mib0, "active": gpu_used_mib()}
    t = time.time()
    engine.release_memory_occupation(tags=["kv_cache", "weights"])
    res["release_s"] = time.time() - t
    time.sleep(2)
    res["mem_mib"]["released"] = gpu_used_mib()
    t = time.time()
    wake_up()
    res["resume_s"] = time.time() - t
    res["mem_mib"]["resumed"] = gpu_used_mib()

    res["phase"] = "cycles"; save()
    res["batch"] = []
    for c in range(args.cycles):
        t = time.time()
        outs = engine.generate(batch_p, {"temperature": 0.6, "top_p": 0.95, "max_new_tokens": 1024,
                                         "stop": ["</sql>", "</solution>"]})
        res["batch"].append({"cycle": c, "s": time.time() - t, "tokens": sum(o["meta_info"]["completion_tokens"] for o in outs)})
        engine.release_memory_occupation(tags=["kv_cache", "weights"])
        wake_up()

    res["phase"] = "greedy_after"; save()
    after = greedy()
    engine.shutdown()
    same = [a["ids"] == b["ids"] for a, b in zip(after, before)]
    res.update(phase="done", greedy_identical=sum(same), greedy_total=len(same),
               verify_ct_before=[b["verify_ct"] for b in before], verify_ct_after=[a["verify_ct"] for a in after],
               accept_before=round(sum(b["tokens"] for b in before) / max(1, sum(b["verify_ct"] for b in before)), 3),
               accept_after=round(sum(a["tokens"] for a in after) / max(1, sum(a["verify_ct"] for a in after)), 3))
    save()
    print("RESULT", json.dumps({k: res[k] for k in ("rl_wakeup", "push_target", "sd", "tp", "mem", "engine", "release_sd_memory", "mem_mib", "greedy_identical",
                                                    "greedy_total", "accept_before", "accept_after", "release_s", "resume_s")}))


if __name__ == "__main__":  # SGLang spawns subprocesses
    main()
