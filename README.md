# EEG-Motor-Imagery-AMSDF
 Code implementation for 7-Class EEG Motor Imagery Decoding using AMSDFNet (BAIC 2026)).

 # 7-Class EEG Motor Imagery Decoding (AMSDFNet)
This repository contains the official PyTorch implementation of the **Adaptive Multi-Scale Disentangled Fusion (AMSDF)** network, presented at the BAIC 2026 conference. 

The model is evaluated on the public Ofner et al. (2017) dataset using a strict 10-fold Leave-One-Run-Out (LORO) cross-validation protocol to prevent data leakage.

### Features
- Physiology-informed filter banks (MRCP, Mu, Beta)
- Cross-band token Transformer
- Strict leakage-free evaluation protocol
