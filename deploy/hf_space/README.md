---
title: Bean Leaf Disease API
emoji: 🌱
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 8000
pinned: false
short_description: MobileNetV2 API for bean leaf disease diagnosis
---

# Bean Leaf Disease API

FastAPI service for the bean leaf disease pipeline (angular leaf spot, bean rust, healthy).
Source code and documentation: https://github.com/dusengepeggy/MLOP-SUMMATIVE

* `GET /docs`: interactive API docs
* `POST /predict`: classify one leaf image
* `POST /upload` and `POST /retrain`: add labelled data and retrain the model
* `GET /status`, `GET /metrics`, `GET /retrain/status`: monitoring

Retrained models are versioned in a Hugging Face model repo (see `model_registry` in `/status`).
