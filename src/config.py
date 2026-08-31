!pip install -q xgboost scikit-learn pandas matplotlib numpy tensorflow

from google.colab import drive
drive.mount('/content/drive')

import os
import gc
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

# 1. GENEL AYARLAR

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

print("TensorFlow:", tf.__version__)
print("GPU:", tf.config.list_physical_devices("GPU"))

CACHE_PATH = "/content/drive/MyDrive/SIU_SLEEP/processed_dataset_final.npz"
RESULT_DIR = "/content/drive/MyDrive/SIU_SLEEP_BALANCED_FINAL_RESULTS"
os.makedirs(RESULT_DIR, exist_ok=True)

CLASS_NAMES = ["W", "N1", "N2", "N3", "REM"]
LABEL_TO_NAME = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
N_CLASSES = 5

TARGET_CHANNELS = [
    "EEG Fpz-Cz",
    "EEG Pz-Oz",
    "EOG horizontal",
    "EMG submental"
]

BATCH_SIZE = 128
EPOCHS = 40
LEARNING_RATE = 2e-4

CLASS_WEIGHT = {
    0: 0.95,  # W
    1: 1.18,  # N1
    2: 1.00,  # N2
    3: 1.13,  # N3
    4: 1.20   # REM
}

RUN_ML_MODELS = True
RUN_MLP = True
RUN_CNN_1D = True
RUN_BILSTM = True
RUN_CNN_BILSTM = True
RUN_ADAPTIVE_FUSION = True

