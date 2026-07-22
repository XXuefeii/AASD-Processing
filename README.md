# AASD Processing  (AASD: An Open Non-Invasive EEG Dataset for Spontaneous Auditory Attention Switch Decoding)

MATLAB and EEGLAB processing utilities for the **Auditory Attention Switching Dataset (AASD)** and the associated study on EEG-based spontaneous auditory attention switch decoding.

---

## 📚 Overview

The **Auditory Attention Switching Dataset (AASD)** is a publicly available EEG dataset developed to investigate cue-free and self-initiated auditory attention switching between two concurrent spatialized speech streams.

The dataset and its experimental paradigm were previously described and released in the data paper:

> **An Open Non-Invasive EEG Dataset for Spontaneous Auditory Attention Switch Decoding**

The associated methodological study uses this previously published dataset to develop and evaluate the **Multi-Resolution Dual-Path EEG Network with Memory-Augmented Contrastive Learning (MAMR-DPNet)**.


---

## 🚀 Dataset Download

The dataset is available through Zenodo for easy access.
### Download the Dataset
AASD on Zenodo (DOI:https://doi.org/10.5281/zenodo.17413336)

---

## 📝 Data Processing

EEG Processing: `src/AASD_processing.m`

### Prerequisites

- **MATLAB** R2021b+ (recommended)
- Toolboxes: **Signal Processing Toolbox** (recommended)
- For CNT import & EEG I/O:
  - **EEGLAB** (recommended) — `loadcnt`, channel ops, event parsing

---

## Dataset Summary

| Item | Description |
|---|---|
| Dataset | Auditory Attention Switching Dataset |
| Abbreviation | AASD |
| Participants | 18 healthy adults |
| Age range | 18–27 years |
| Language | Mandarin Chinese |
| Scalp EEG electrodes | 64 |
| Raw sampling rate | 500 Hz |
| Trials per participant | 60 spontaneous attention-switching trials |
| Trial duration | 60 s |
| Experimental organization | 6 blocks × 10 trials |
| Attention states | Left or right auditory attention |
| Auditory scene | Two concurrent spatialized speech streams |
| Spatial positions | −90° and +90° |
| Speech materials | AISHELL-1 Mandarin speech corpus |
| Speakers | Three male and three female speakers |
| Behavioral reports | Participant keypresses marking self-initiated switches |

During each trial, participants listened to two concurrent spatialized speech streams and voluntarily switched their attention between them at self-selected times.

No countdown, visual cue, auditory cue, or other time-locked prompt instructed participants when to switch.

---

## Repository Structure

```text
AASD-Processing/
├── README.md
└── src/
    └── AASD_processing.m

---
 
## 🔐 Ethics & Privacy

IRB: SUSTech No. 2022DZX003

Written informed consent; public release of anonymized data


