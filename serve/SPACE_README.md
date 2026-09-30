---
title: Seam Deepfake Forensics API
colorFrom: gray
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

API behind the Seam demo (https://github.com/Krupal-1219/Deepfake-detection-SSL).

- `POST /analyze` with a multipart `file` (JPEG, PNG or WebP, up to 10 MB)
- `GET /health`, `GET /info`

Uploads are processed in memory and never stored.
