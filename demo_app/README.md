---
title: NVDA Volatility Demo
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# NVDA Volatility Presentation

Dash/Plotly presentation app for the OTPP take-home project. The app reads
committed presentation snapshots from `assets/` and does not retrain models,
call APIs, or depend on ignored processed-data folders.

The Docker Space serves `app:server` through Gunicorn on port `7860`.
