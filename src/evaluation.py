from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, classification_report,
    cohen_kappa_score, confusion_matrix, f1_score, precision_score, recall_score,
)

from .config import CLASS_NAMES, N_CLASSES, TARGET_CHANNELS

LABELS = list(range(N_CLASSES))


def evaluate_predictions(y_true, y_pred, model_name, group_name):
    report = classification_report(y_true, y_pred, labels=LABELS,
                                   target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    # Explicit labels keep the metric a five-class average even if a split lacks a class.
    result = {
        "Grup": group_name, "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro Precision": precision_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "Macro Recall": recall_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "Macro F1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "Weighted F1": f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0),
        "Kappa": cohen_kappa_score(y_true, y_pred, labels=LABELS),
    }
    for name in CLASS_NAMES:
        result[f"{name} F1"] = report[name]["f1-score"]
        if name in ("N1", "N2", "N3"):
            result[f"{name} Precision"] = report[name]["precision"]
            result[f"{name} Recall"] = report[name]["recall"]
    return result


class Reporter:
    def __init__(self, output_dir, *, show_plots=False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.show_plots = show_plots

    def _save(self, fig, filename):
        fig.tight_layout()
        fig.savefig(self.output_dir / filename, dpi=300, bbox_inches="tight")
        if self.show_plots:
            plt.show()
        plt.close(fig)

    def predictions(self, y_true, y_pred, model_name, *, probabilities=None):
        filename = model_name.replace(" ", "_")
        report = classification_report(y_true, y_pred, labels=LABELS,
                                       target_names=CLASS_NAMES, zero_division=0)
        (self.output_dir / f"{filename}_classification_report.txt").write_text(report, encoding="utf-8")
        predictions = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
        if probabilities is not None:
            for index, name in enumerate(CLASS_NAMES):
                predictions[f"p_{name}"] = probabilities[:, index]
        predictions.to_csv(self.output_dir / f"{filename}_predictions.csv", index=False)
        for normalize, suffix in ((None, "counts"), ("true", "normalized")):
            matrix = confusion_matrix(y_true, y_pred, labels=LABELS, normalize=normalize)
            pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
                self.output_dir / f"{filename}_confusion_{suffix}.csv")
        fig, ax = plt.subplots(figsize=(7, 6))
        ConfusionMatrixDisplay(matrix, display_labels=CLASS_NAMES).plot(
            ax=ax, cmap="Blues", values_format=".2f", colorbar=True)
        ax.set(title=f"{model_name} Normalize Karışıklık Matrisi",
               xlabel="Tahmin Edilen Etiket", ylabel="Gerçek Etiket")
        ax.images[-1].colorbar.set_label("Oran")
        self._save(fig, f"{filename}_normalize_karisiklik_matrisi.png")
        print(report)

    def history(self, history, model_name):
        pd.DataFrame(history.history).to_csv(self.output_dir / f"{model_name}_history.csv", index=False)
        for metric, label, suffix in (("accuracy", "Doğruluk", "dogruluk"), ("loss", "Kayıp", "kayip")):
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(history.history[metric], label="Eğitim")
            ax.plot(history.history[f"val_{metric}"], label="Doğrulama")
            ax.set(xlabel="Epok", ylabel=label, title=f"{model_name} Eğitim ve Doğrulama {label}")
            ax.legend()
            self._save(fig, f"{model_name}_{suffix}_egri.png")

    def channel_weights(self, weights):
        mean_weights = np.mean(weights.squeeze(-1), axis=0)
        if len(mean_weights) != len(TARGET_CHANNELS):
            raise ValueError("Attention channel count does not match TARGET_CHANNELS.")
        frame = pd.DataFrame({"Kanal": TARGET_CHANNELS, "Ortalama Dikkat Ağırlığı": mean_weights})
        frame.to_csv(self.output_dir / "adaptive_channel_weights.csv", index=False, encoding="utf-8-sig")
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(frame["Kanal"], frame["Ortalama Dikkat Ağırlığı"])
        ax.set(xlabel="PSG Kanalı", ylabel="Ortalama Dikkat Ağırlığı",
               title="Balanced Adaptive Fusion Kanal Ağırlıkları")
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
        self._save(fig, "adaptive_channel_weights.png")

    def comparison(self, results, *, synthetic=False):
        frame = pd.DataFrame(results)
        if frame.empty:
            raise ValueError("No model results to report.")
        columns = ["Grup", "Model", "Accuracy", "Macro Precision", "Macro Recall", "Macro F1",
                   "Weighted F1", "Kappa", "N1 Precision", "N1 Recall", "N1 F1", "N2 Precision",
                   "N2 Recall", "N2 F1", "W F1", "N3 F1", "REM F1", "N3 Precision", "N3 Recall"]
        frame = frame[columns]
        frame.to_csv(self.output_dir / "tum_model_karsilastirma_sonuclari.csv", index=False, encoding="utf-8-sig")
        metrics = ["Accuracy", "Macro F1", "Kappa", "N1 F1", "N2 F1"]
        filenames = ["accuracy", "macro_f1", "kappa", "n1_f1", "n2_f1"]
        for metric, filename in zip(metrics, filenames):
            fig, ax = plt.subplots(figsize=(12, 5.5))
            ax.bar(frame["Model"], frame[metric])
            ax.set(xlabel="Model", ylabel=metric, title=f"Tüm Modellerin {metric} Karşılaştırması",
                   ylim=(-1 if metric == "Kappa" else 0, 1.05))
            plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
            self._save(fig, f"{filename}_tum_modeller.png")
        fig, ax = plt.subplots(figsize=(14, 6))
        x, width = np.arange(len(frame)), 0.15
        for index, metric in enumerate(metrics):
            ax.bar(x + index * width, frame[metric], width, label=metric)
        ax.set_xticks(x + width * 2, frame["Model"], rotation=25, ha="right")
        ax.set(ylabel="Metrik Değeri", title="Tüm Modellerin Dengeli Karşılaştırılması")
        ax.legend()
        self._save(fig, "tum_modeller_tum_metrikler_karsilastirma.png")
        lines = ["SYNTHETIC SOFTWARE CHECK — NOT RESEARCH RESULTS" if synthetic else "ÖZET SONUÇLAR"]
        for metric in metrics:
            valid = frame.dropna(subset=[metric])
            if valid.empty:
                lines.append(f"{metric}: undefined for this test split")
                continue
            best = valid.sort_values(metric, ascending=False).iloc[0]
            lines.append(f"{metric}: {best['Model']} | {best[metric]:.4f}")
        lines += ["", "Best checkpoint: 0.45 MacroF1 + 0.25 N2F1 + 0.15 N1F1 + 0.10 REMF1 + 0.05 N3F1.",
                  "No focal loss. Test-set rankings are descriptive, not validation-based model selection."]
        (self.output_dir / "ozet_sonuclar.txt").write_text("\n".join(lines), encoding="utf-8")
        return frame
