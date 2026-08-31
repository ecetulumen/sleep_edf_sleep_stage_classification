# 5. METRİK VE GRAFİK FONKSİYONLARI

def evaluate_predictions(y_true, y_pred, model_name, group_name):
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2, 3, 4],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    return {
        "Grup": group_name,
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Macro Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "Macro F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Weighted F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "Kappa": cohen_kappa_score(y_true, y_pred),
        "W F1": report["W"]["f1-score"],
        "N1 Precision": report["N1"]["precision"],
        "N1 Recall": report["N1"]["recall"],
        "N1 F1": report["N1"]["f1-score"],
        "N2 Precision": report["N2"]["precision"],
        "N2 Recall": report["N2"]["recall"],
        "N2 F1": report["N2"]["f1-score"],
        "N3 Precision": report["N3"]["precision"],
        "N3 Recall": report["N3"]["recall"],
        "N3 F1": report["N3"]["f1-score"],
        "REM F1": report["REM"]["f1-score"]
    }


def save_classification_report(y_true, y_pred, model_name):
    text = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2, 3, 4],
        target_names=CLASS_NAMES,
        zero_division=0
    )

    path = os.path.join(RESULT_DIR, f"{model_name}_classification_report.txt")

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

    print("\n" + "=" * 80)
    print(model_name)
    print("=" * 80)
    print(text)


def plot_confusion_matrix_tr(y_true, y_pred, model_name):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1, 2, 3, 4],
        normalize="true"
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_NAMES
    )

    disp.plot(
        ax=ax,
        cmap="Blues",
        values_format=".2f",
        colorbar=True
    )

    ax.set_title(f"{model_name} Normalize Karışıklık Matrisi", fontsize=13)
    ax.set_xlabel("Tahmin Edilen Etiket", fontsize=12)
    ax.set_ylabel("Gerçek Etiket", fontsize=12)
    ax.images[-1].colorbar.set_label("Oran", fontsize=11)

    plt.tight_layout()

    path = os.path.join(RESULT_DIR, f"{model_name}_normalize_karisiklik_matrisi.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_history(history, model_name):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(history.history["accuracy"], label="Eğitim Doğruluğu")
    ax.plot(history.history["val_accuracy"], label="Doğrulama Doğruluğu")
    ax.set_xlabel("Epok")
    ax.set_ylabel("Doğruluk")
    ax.set_title(f"{model_name} Eğitim ve Doğrulama Doğruluğu")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(RESULT_DIR, f"{model_name}_dogruluk_egri.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(history.history["loss"], label="Eğitim Kaybı")
    ax.plot(history.history["val_loss"], label="Doğrulama Kaybı")
    ax.set_xlabel("Epok")
    ax.set_ylabel("Kayıp")
    ax.set_title(f"{model_name} Eğitim ve Doğrulama Kaybı")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(RESULT_DIR, f"{model_name}_kayip_egri.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_metric_bar(df, metric, filename, title):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(df["Model"], df[metric])
    ax.set_xlabel("Model")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    path = os.path.join(RESULT_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_all_metrics(df):
    metrics = ["Accuracy", "Macro F1", "Kappa", "N1 F1", "N2 F1"]
    x = np.arange(len(df["Model"]))
    width = 0.15

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, df[metric], width, label=metric)

    ax.set_xlabel("Model")
    ax.set_ylabel("Metrik Değeri")
    ax.set_title("Tüm Modellerin Dengeli Karşılaştırılması")
    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(df["Model"], rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()
    plt.tight_layout()

    path = os.path.join(RESULT_DIR, "tum_modeller_tum_metrikler_karsilastirma.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()


def clear_memory():
    tf.keras.backend.clear_session()

