#!/bin/bash
# Phases 2, 4 and 6 in one detached run: install, serve, verify the API,
# smoke benchmark. Also records where the model weights actually land on
# disk, which a naive directory walk gets wrong (the HF hub cache uses
# symlinks from snapshots/ into blobs/).
exec > ~/phase46.log 2>&1
set -x

cd ~
./setup.sh > ~/setup.log 2>&1; echo "SETUP_EXIT=$?"
tail -5 ~/setup.log

source $HOME/.local/bin/env; source ~/uv_venv/bin/activate
export UV_TORCH_BACKEND=cpu
uv pip install -q openai && echo "OPENAI_OK"

echo "###### disk before download ######"
df -h / | tail -1
du -sh ~/.cache/huggingface 2>/dev/null || echo "no hf cache yet"

echo "###### starting server ######"
nohup ./serve.sh > ~/serve_api.log 2>&1 &
for i in $(seq 1 400); do
  curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 && { echo "SERVER_UP_AFTER=$((i*2))s"; break; }
  sleep 2
done

echo "###### where did the weights go? ######"
df -h / | tail -1
echo "--- du of each hf subdir (du does NOT follow symlinks) ---"
du -sh ~/.cache/huggingface/* 2>/dev/null
echo "--- files over 100MB anywhere under the cache ---"
find ~/.cache/huggingface -type f -size +100M 2>/dev/null | head -10
echo "--- apparent size following symlinks ---"
du -shL ~/.cache/huggingface/hub/models--Qwen--Qwen3-8B 2>/dev/null
echo "--- snapshot safetensors entries ---"
ls -l ~/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/*/ 2>/dev/null | grep -i safetensors | head -6

echo "###### tpu-info with server up ######"
tpu-info 2>&1 | grep -B2 -A6 "HBM Usage"

echo "###### startup facts from the log ######"
grep -oE "compile_sizes.: \[[0-9, ]+\]" ~/serve_api.log | head -1
grep -cE "compile_prefix isolation" ~/serve_api.log | sed 's/^/compile_prefix_warnings=/'
grep -oE "Automatically using fp8_[a-z0-9]+ for FP8 KV cache on TPU v6e" ~/serve_api.log | head -1
grep -iE "KV cache size|Maximum concurrency|available KV cache|model weights take|GPU KV cache" ~/serve_api.log | head -6

echo "###### PHASE 4: API endpoints ######"
python smoke_test.py --out results/smoke_test_results.json
echo "SMOKE_EXIT=$?"

echo "###### PHASE 6: smoke benchmark ######"
bash benchmark_smoke.sh
echo "BENCH_EXIT=$?"

echo "###### cache dirs ######"
du -sh ~/vllm_torchtpu_cache/* 2>/dev/null

echo "###### shutdown ######"
pkill -f "vllm serve"; sleep 5
echo "RUNALL_DONE"
