"""Per-step comparison of verl console logs (lines "step:N - key:value - ...").

Usage: python compare_runs.py <a.log> <b.log> [...]   (first log is the reference for the speedup columns)
"""

import re
import sys

KEYS = [
    ("timing_s/gen", "gen s"),
    ("timing_s/update_actor", "update s"),
    ("timing_s/step", "step s"),
    ("critic/score/mean", "score"),
    ("response_length/mean", "resp mean"),
    ("response_length/max", "resp max"),
]
LINE = re.compile(r"\bstep:(\d+) - (.*)")
KV = re.compile(r"([\w/@.-]+):(-?[\d.]+(?:e[-+]?\d+)?|nan|inf)")


def parse(path):
    steps = {}
    for line in open(path, errors="replace"):
        m = LINE.search(line)
        if m:
            steps[int(m.group(1))] = {k: float(v) for k, v in KV.findall(m.group(2))}
    return steps


def main():
    runs = [(p, parse(p)) for p in sys.argv[1:]]
    for path, steps in runs:
        print(f"\n{path}: {len(steps)} steps")
        print("step " + " ".join(f"{h:>10}" for _, h in KEYS))
        for s in sorted(steps):
            print(f"{s:>4} " + " ".join(f"{steps[s].get(k, float('nan')):>10.3f}" for k, _ in KEYS))
    if len(runs) >= 2:
        (pa, a), (pb, b) = runs[0], runs[1]
        common = sorted(set(a) & set(b))
        print(f"\n{len(common)} common steps; ratio = {pb} / {pa} (>1 means the first run is faster)")
        for k, h in KEYS[:3]:
            sa, sb = sum(a[s].get(k, 0) for s in common), sum(b[s].get(k, 0) for s in common)
            print(f"  {h:>10}: {sa:9.1f} vs {sb:9.1f}  ratio {sb / sa if sa else float('nan'):.3f}")
        if common:
            s = common[0]
            print(f"  step {s} only (same prompts in both arms): gen {a[s].get('timing_s/gen', 0):.1f} vs "
                  f"{b[s].get('timing_s/gen', 0):.1f}")


if __name__ == "__main__":
    main()
