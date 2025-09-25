# Deep Learning for Electrocardiogram Analysis 

In this research, we seek to predict Cancer Therapy Related Cardiac Dysfunction (CTRCD) in cancer patients using ECG waveform data. The study-group of cancer patients received Immune Checkpoint Inhibitor (ICI) Immunotherapy and conducted ECG tests. The data is privately stored with UCLA health and includes over 30,000 patients ECG results.

This project applies machine learning and deep neural networks to ECG waveforms and clinical features to predict CTRCD. It explores both tabular (Random Forests with bootstrap validation) and time-series models (1D CNNs + multimodal Two-Tower networks), addressing challenges of limited positives, high variance, and noisy data. The modular codebase demonstrates a full ML pipeline — from ECG parsing to augmentation, model training, and evaluation — and can be adapted for other biomedical signal tasks.
