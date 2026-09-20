# GKE benchmark image: torch (CUDA) + DeepSpeed + transformers
# Build:  gcloud builds submit --tag $IMAGE .
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

RUN pip install --no-cache-dir \
      "deepspeed>=0.18.2" \
      "transformers>=4.56,<5.0.0" \
      accelerate nvidia-ml-py sentencepiece protobuf google-cloud-storage

WORKDIR /bench
COPY common.py bench_zero_stages.py bench_fsdp.py summarize.py ./
COPY ds_configs ./ds_configs
COPY gke/upload_results.py ./upload_results.py
