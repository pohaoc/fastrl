"""Preflight for the eagle-train env: imports (DeepSpeed needs a GPU node: its Triton kernels look for a
driver at import, and transformers imports DeepSpeed when it is installed) and the mapping of the released
drafter's tensors onto eagle-train's model (what --init_draft_path loads). Run from eagle-train/."""
import sys

import deepspeed
import flash_attn
import torch
import transformers
from huggingface_hub import snapshot_download
from transformers import AutoConfig

sys.path.insert(0, ".")
from model.qwen2_eagle import Qwen2ForCausalLMEagle  # noqa: E402

print("torch", torch.__version__, "transformers", transformers.__version__, "deepspeed", deepspeed.__version__,
      "flash_attn", flash_attn.__version__, "cuda", torch.cuda.is_available())
target, init = sys.argv[1], sys.argv[2]
cfg = AutoConfig.from_pretrained(snapshot_download(target))
cfg.num_hidden_layers = 1
with torch.device("meta"):
    m = Qwen2ForCausalLMEagle(cfg)
keys = {k: tuple(v.shape) for k, v in m.state_dict().items()}
sd = torch.load(snapshot_download(init) + "/pytorch_model.bin", map_location="cpu", mmap=True, weights_only=True)
ck = {(k if k.startswith(("model.", "lm_head")) else "model." + k): tuple(v.shape) for k, v in sd.items()}
missing, unexpected = sorted(set(keys) - set(ck)), sorted(set(ck) - set(keys))
bad = sorted(k for k in set(keys) & set(ck) if keys[k] != ck[k])
print("missing from checkpoint:", missing, "| unexpected:", unexpected, "| shape mismatches:", bad)
assert not unexpected and not bad and set(missing) <= {"lm_head.weight"}, "drafter checkpoint does not map"
print("preflight OK")
