# Hugging Face Space (Docker, CPU basic) for the Seam deepfake-forensics API.
# Space repo layout: Dockerfile, README.md, requirements.txt, serve/ (the package)
FROM python:3.11-slim

RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PYTHONUNBUFFERED=1 \
    TORCH_HOME=/home/user/models/torch \
    HF_HOME=/home/user/models/hf \
    DEEPFAKE_WEIGHTS=/home/user/models/best.pt \
    WEIGHTS_REPO=gvrathodd/seam-weights
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

USER user

# Bake the model code and the public AI-image detector into the image so cold starts are faster.
RUN python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg', pretrained=False)" \
 && python -c "from transformers import AutoImageProcessor, SiglipForImageClassification as S; m='Ateeqq/ai-vs-human-image-detector'; AutoImageProcessor.from_pretrained(m); S.from_pretrained(m)"

# The trained checkpoint lives in a private model repo; HF_TOKEN is a Space secret.
RUN --mount=type=secret,id=HF_TOKEN,mode=0444,required=true \
    python -c "import os; from huggingface_hub import hf_hub_download; hf_hub_download(os.environ['WEIGHTS_REPO'], 'best.pt', local_dir='/home/user/models', token=open('/run/secrets/HF_TOKEN').read().strip())"

COPY --chown=user serve ./serve

EXPOSE 7860
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "7860"]
