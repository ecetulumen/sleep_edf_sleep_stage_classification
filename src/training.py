# 6. DENGELİ CALLBACK

class BalancedF1Callback(tf.keras.callbacks.Callback):
    """
    En iyi epoch'u sadece val_accuracy ile değil,
    Macro F1 + N2 F1 + N1 F1 + REM F1 dengesine göre seçer.

    N2'nin çok düşmesini engellemek için N2'ye ağırlık verildi.
    """

    def __init__(self, val_data, y_val, model_name, patience=8):
        super().__init__()
        self.val_data = val_data
        self.y_val = y_val
        self.model_name = model_name
        self.patience = patience
        self.best_score = -np.inf
        self.wait = 0
        self.best_path = os.path.join(RESULT_DIR, f"{model_name}_best_balanced.weights.h5")

    def on_epoch_end(self, epoch, logs=None):
        y_prob = self.model.predict(self.val_data, batch_size=BATCH_SIZE, verbose=0)
        y_pred = np.argmax(y_prob, axis=1)

        macro_f1 = f1_score(self.y_val, y_pred, average="macro", zero_division=0)

        f1_each = f1_score(
            self.y_val,
            y_pred,
            labels=[0, 1, 2, 3, 4],
            average=None,
            zero_division=0
        )

        w_f1, n1_f1, n2_f1, n3_f1, rem_f1 = f1_each

        # N2 ve REM'i korurken N1'i de tamamen bırakmayan skor
        balanced_score = (
            0.45 * macro_f1 +
            0.25 * n2_f1 +
            0.15 * n1_f1 +
            0.10 * rem_f1 +
            0.05 * n3_f1
        )

        print(
            f"\nVal MacroF1={macro_f1:.4f} | "
            f"N1F1={n1_f1:.4f} | N2F1={n2_f1:.4f} | "
            f"N3F1={n3_f1:.4f} | REMF1={rem_f1:.4f} | "
            f"BalancedScore={balanced_score:.4f}"
        )

        if balanced_score > self.best_score:
            self.best_score = balanced_score
            self.wait = 0
            self.model.save_weights(self.best_path)
            print(f"Yeni en iyi ağırlık kaydedildi: {self.best_score:.4f}")
        else:
            self.wait += 1
            print(f"İyileşme yok: {self.wait}/{self.patience}")

            if self.wait >= self.patience:
                print("Balanced callback early stopping.")
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        if os.path.exists(self.best_path):
            self.model.load_weights(self.best_path)
            print(f"En iyi dengeli ağırlıklar geri yüklendi: {self.best_score:.4f}")


def get_callbacks(model_name, val_data):
    return [
        BalancedF1Callback(
            val_data=val_data,
            y_val=y_val,
            model_name=model_name,
            patience=8
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]
# 8. DL TRAIN FUNCTION

def train_and_evaluate_dl(
    model,
    train_data,
    val_data,
    test_data,
    model_name,
    use_class_weight=True
):
    print("\n" + "=" * 80)
    print(f"{model_name} eğitiliyor...")
    print("=" * 80)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    callbacks = get_callbacks(model_name, val_data)

    fit_kwargs = {
        "x": train_data,
        "y": y_train,
        "validation_data": (val_data, y_val),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "callbacks": callbacks,
        "verbose": 1
    }

    if use_class_weight:
        fit_kwargs["class_weight"] = CLASS_WEIGHT

    history = model.fit(**fit_kwargs)

    y_prob = model.predict(test_data, batch_size=BATCH_SIZE)
    y_pred = np.argmax(y_prob, axis=1)

    result = evaluate_predictions(y_test, y_pred, model_name, "Deep Learning")

    save_classification_report(y_test, y_pred, model_name)
    plot_history(history, model_name)
    plot_confusion_matrix_tr(y_test, y_pred, model_name)

    return result, y_pred, y_prob, history, model

