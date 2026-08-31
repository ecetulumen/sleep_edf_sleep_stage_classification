# Sleep-EDF-Sleep-Stage-Classification
# Sleep Stage Classification from Multi-Channel PSG Signals

This project focuses on automatic sleep stage classification from multi-channel polysomnography (PSG) signals using machine learning and deep learning approaches.

The main goal is to classify five sleep stages — **W, N1, N2, N3, and REM** — while considering class imbalance and the difficulty of transitional stages such as N1.

## Dataset

The project uses the **Sleep-EDF Sleep Cassette (SC)** dataset.

The PSG signals include:

- EEG Fpz-Cz
- EEG Pz-Oz
- EOG horizontal
- EMG submental

Each recording is segmented into **30-second epochs** and labeled into five sleep stages:

- W
- N1
- N2
- N3
- REM

## Methodology

The pipeline includes:

1. Multi-channel PSG data preparation
2. Subject-wise train / validation / test split
3. Feature scaling
4. Machine learning and deep learning model training
5. Class-weighted learning
6. Performance evaluation using multiple metrics
7. Adaptive channel fusion analysis

## Models

The following models were evaluated:

### Machine Learning
- Random Forest
- Support Vector Machine (SVM)
- XGBoost

### Deep Learning
- Multi-Layer Perceptron (MLP)
- 1D CNN
- BiLSTM with Attention
- CNN-BiLSTM
- CNN-BiLSTM with Adaptive Fusion

## Adaptive Fusion

The proposed Adaptive Fusion architecture learns the contribution of each PSG channel instead of treating all channels equally.

The model combines:

- CNN-based local feature extraction
- BiLSTM-based temporal modeling
- Attention mechanisms
- Learned channel weighting
- Fully connected classification layers

## Evaluation Metrics

The models are evaluated using:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1-score
- Weighted F1-score
- Cohen's Kappa
- Class-wise F1-scores

Special attention is given to the performance of difficult classes such as N1 and REM.

## Results

The final Adaptive Fusion model achieved:

- **Accuracy:** 82.62%
- **Macro F1:** 75.66%
- **Cohen's Kappa:** 75.51%

The normalized confusion matrix shows strong classification performance for W, N2, and N3, while N1 remains the most challenging stage.

### Training Performance

![Accuracy Curve](results/adaptive_fusion_accuracy_curve.png)

![Loss Curve](results/adaptive_fusion_loss_curve.png)

### Confusion Matrix

![Confusion Matrix](results/final_adaptive_fusion_confusion_matrix.png)

### Adaptive Channel Weights

![Adaptive Channel Weights](results/adaptive_channel_weights.png)

### Dataset Distribution

![Sleep Stage Duration](results/final_uyku_evreleri_toplam_sure_saat.png)

![Dataset Summary](results/final_veri_ozet_tablo.png)

## Project Structure

```text
sleep_edf_sleep_stage_classification/
│
├── src/
│   ├── config.py
│   ├── data_utils.py
│   ├── models.py
│   ├── training.py
│   ├── evaluation.py
│   └── main.py
│
├── notebooks/
├── results/
├── requirements.txt
├── README.md
└── .gitignore
