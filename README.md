# RSNA Intracranial Hemorrhage Classification with MONAI

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![MONAI](https://img.shields.io/badge/MONAI-1.3.0-blue)](https://monai.io/)

## Project Overview

This repository contains a deep learning pipeline for classifying intracranial hemorrhage types from CT scans using MONAI framework. The system:
- Preprocesses DICOM images into standardized PNG format
- Trains a DenseNet121 model on 500 samples per hemorrhage class
- Evaluates model performance on validation data

## Dataset

The RSNA Intracranial Hemorrhage Detection dataset from Kaggle:
https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection

## Installation

```bash
git clone https://github.com/yourusername/rsna-hemorrhage-classification.git
cd rsna-hemorrhage-classification
pip install -r requirements.txt

