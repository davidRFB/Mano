---
title: Mano LSC Translator
emoji: 🤟
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Mano LSC Translator

This is the API backend for the Colombian Sign Language Translator.


# INSTALL  (TODO: Fix with requierements)
micromamba create -n Mano python=3.11
micromamba activate Mano
micromamba install pytorch torchvision -c pytorch -c conda-forge
pip install opencv-python --break-system-packages
pip install mediapipe==0.10.14 --break-system-packages # mediapipe remove solutions for hand detection

