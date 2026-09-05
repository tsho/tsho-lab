# Runbook — TPU v6e (FLEX_START) で回す完全手順

git pull から VM 削除まで。所要 ~30-60分 (キュー待ち除く)。
FLEX_START = キュー式確保: 要求を積むと容量が空き次第プロビジョニングされ、
`--max-run-duration` 経過で自動 DELETE (消し忘れ事故が構造的に起きない)。

## Phase 0: コード取得 (Cloud Shell)

```bash
cd ~/tsho-lab
git fetch origin && git checkout feature/jax-sharding-bench && git pull
ls jax-sharding-bench/          # bench.py / README.md
```

## Phase 1: TPU VM 作成 (FLEX_START)

推奨ゾーン (Builder program ガイダンス): v6e は us-east5-a / us-east5-b /
us-central1-a / europe-west4-a。マシンタイプ ct6e-standard-4t = v6e ×4チップ。

```bash
gcloud compute instances create tpu-v6e-vm \
    --project=YOUR_PROJECT \
    --zone=us-east5-a \
    --machine-type=ct6e-standard-4t \
    --provisioning-model=FLEX_START \
    --request-valid-for-duration=2h \
    --max-run-duration=4h \
    --instance-termination-action=DELETE \
    --image-project=ubuntu-os-accelerator-images \
    --image-family=ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e \
    --maintenance-policy=TERMINATE \
    --metadata=startup-script="echo 'TPU VM Booted'"

# キュー待ち監視 (RUNNING が出たら Ctrl-C)
watch -n 30 "gcloud compute instances describe tpu-v6e-vm --zone=us-east5-a \
  --project=YOUR_PROJECT --format='value(status)' 2>/dev/null || echo QUEUED"
```

2時間で容量が来なければ要求は失効 → zone を変えて再投入。

## Phase 2: コード転送 & SSH

```bash
gcloud compute scp ~/tsho-lab/jax-sharding-bench/bench.py tpu-v6e-vm:~/ \
    --zone=us-east5-a --project=YOUR_PROJECT
gcloud compute ssh tpu-v6e-vm --zone=us-east5-a --project=YOUR_PROJECT
```

## Phase 3: セットアップ & 実測 (VM 内)

```bash
sudo apt-get update -qq && sudo apt-get install -y -qq python3-pip python3-venv
python3 -m venv ~/jaxenv && source ~/jaxenv/bin/activate
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
python3 -c "import jax; print(jax.default_backend(), len(jax.devices()))"   # → "tpu 4"

python3 bench.py --mode dp   --steps 15
python3 bench.py --mode fsdp --steps 15
python3 bench.py --mode dp   --dim 3072 --layers 24 --steps 15
python3 bench.py --mode fsdp --dim 3072 --layers 24 --steps 15
cat results/*.jsonl
exit
```

## Phase 4: 結果回収 → 削除

```bash
gcloud compute scp --recurse tpu-v6e-vm:~/results ~/tpu-results \
    --zone=us-east5-a --project=YOUR_PROJECT
gcloud compute instances delete tpu-v6e-vm \
    --zone=us-east5-a --project=YOUR_PROJECT --quiet
```

## トラブルシュート

| 症状 | 対処 |
|---|---|
| 2h 待って失効 | zone 替え (us-east5-b → us-central1-a → europe-west4-a) |
| `cpu` と表示 (libtpu 未認識) | venv で `pip install "jax[tpu]"` の -f URL を確認 / VM 再起動 |
| OOM | まず --dim を下げる。fsdp は dp よりモデル上限が高いはず (それ自体が測定結果) |
