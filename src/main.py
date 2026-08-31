# 9. MODEL EĞİTİMLERİ

all_results = []
all_predictions = {}

# -------------------------
# ML MODELLERİ
# -------------------------

if RUN_ML_MODELS:
    print("\n" + "=" * 80)
    print("MAKİNE ÖĞRENMESİ MODELLERİ")
    print("=" * 80)

    rf_model = RandomForestClassifier(
        n_estimators=500,
        max_depth=20,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        class_weight=CLASS_WEIGHT,
        n_jobs=-1
    )

    rf_model.fit(X_ml_trainval, y_ml_trainval)
    rf_pred = rf_model.predict(Xfc_test_scaled)

    all_results.append(evaluate_predictions(y_test, rf_pred, "Random Forest", "Makine Öğrenmesi"))
    all_predictions["Random Forest"] = rf_pred
    save_classification_report(y_test, rf_pred, "Random_Forest")
    plot_confusion_matrix_tr(y_test, rf_pred, "Random_Forest")

    del rf_model
    gc.collect()

    svm_model = SVC(
        kernel="rbf",
        C=3.0,
        gamma="scale",
        class_weight=CLASS_WEIGHT,
        probability=False,
        random_state=RANDOM_STATE
    )

    svm_model.fit(X_ml_trainval, y_ml_trainval)
    svm_pred = svm_model.predict(Xfc_test_scaled)

    all_results.append(evaluate_predictions(y_test, svm_pred, "SVM", "Makine Öğrenmesi"))
    all_predictions["SVM"] = svm_pred
    save_classification_report(y_test, svm_pred, "SVM")
    plot_confusion_matrix_tr(y_test, svm_pred, "SVM")

    del svm_model
    gc.collect()

    xgb_model = XGBClassifier(
        objective="multi:softmax",
        num_class=N_CLASSES,
        n_estimators=400,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=2.5,
        reg_alpha=0.3,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    xgb_model.fit(
        X_ml_trainval,
        y_ml_trainval,
        sample_weight=sample_weight_ml
    )

    xgb_pred = xgb_model.predict(Xfc_test_scaled)

    all_results.append(evaluate_predictions(y_test, xgb_pred, "XGBoost", "Makine Öğrenmesi"))
    all_predictions["XGBoost"] = xgb_pred
    save_classification_report(y_test, xgb_pred, "XGBoost")
    plot_confusion_matrix_tr(y_test, xgb_pred, "XGBoost")

    del xgb_model
    gc.collect()

# DL MODELLERİ

raw_shape = Xraw_train.shape[1:]
seq_shape = Xfs_train_scaled.shape[1:]
feature_dim = Xfc_train_scaled.shape[1]

print("\n" + "=" * 80)
print("DEEP LEARNING MODELLERİ")
print("=" * 80)

if RUN_MLP:
    mlp_model = build_mlp(feature_dim)

    res, pred, prob, hist, trained = train_and_evaluate_dl(
        model=mlp_model,
        train_data=Xfc_train_scaled,
        val_data=Xfc_val_scaled,
        test_data=Xfc_test_scaled,
        model_name="MLP",
        use_class_weight=True
    )

    all_results.append(res)
    all_predictions["MLP"] = pred

    del mlp_model, trained, hist, prob
    clear_memory()


if RUN_CNN_1D:
    cnn_model = build_cnn_1d(raw_shape)

    res, pred, prob, hist, trained = train_and_evaluate_dl(
        model=cnn_model,
        train_data=Xraw_train,
        val_data=Xraw_val,
        test_data=Xraw_test,
        model_name="CNN_1D",
        use_class_weight=True
    )

    all_results.append(res)
    all_predictions["CNN_1D"] = pred

    del cnn_model, trained, hist, prob
    clear_memory()


if RUN_BILSTM:
    bilstm_model = build_bilstm(seq_shape)

    res, pred, prob, hist, trained = train_and_evaluate_dl(
        model=bilstm_model,
        train_data=Xfs_train_scaled,
        val_data=Xfs_val_scaled,
        test_data=Xfs_test_scaled,
        model_name="BiLSTM_Attention",
        use_class_weight=True
    )

    all_results.append(res)
    all_predictions["BiLSTM_Attention"] = pred

    del bilstm_model, trained, hist, prob
    clear_memory()


if RUN_CNN_BILSTM:
    cnn_bilstm_model = build_balanced_cnn_bilstm(raw_shape, seq_shape)

    res, pred, prob, hist, trained = train_and_evaluate_dl(
        model=cnn_bilstm_model,
        train_data=[Xraw_train, Xfs_train_scaled],
        val_data=[Xraw_val, Xfs_val_scaled],
        test_data=[Xraw_test, Xfs_test_scaled],
        model_name="Balanced_CNN_BiLSTM",
        use_class_weight=True
    )

    all_results.append(res)
    all_predictions["Balanced_CNN_BiLSTM"] = pred

    del cnn_bilstm_model, trained, hist, prob
    clear_memory()


if RUN_ADAPTIVE_FUSION:
    adaptive_model = build_balanced_adaptive_fusion(
        raw_shape=raw_shape,
        seq_shape=seq_shape,
        feature_dim=feature_dim
    )

    res, pred, prob, hist, trained = train_and_evaluate_dl(
        model=adaptive_model,
        train_data=[Xraw_train, Xfs_train_scaled, Xfc_train_scaled],
        val_data=[Xraw_val, Xfs_val_scaled, Xfc_val_scaled],
        test_data=[Xraw_test, Xfs_test_scaled, Xfc_test_scaled],
        model_name="Balanced_AdaptiveFusion",
        use_class_weight=True
    )

    all_results.append(res)
    all_predictions["Balanced_AdaptiveFusion"] = pred

    # Kanal ağırlıkları
    try:
        attention_model = models.Model(
            inputs=trained.input,
            outputs=trained.get_layer("adaptive_channel_weights").output
        )

        attn = attention_model.predict(
            [Xraw_test, Xfs_test_scaled, Xfc_test_scaled],
            batch_size=BATCH_SIZE,
            verbose=0
        )

        mean_attn = np.mean(attn.squeeze(-1), axis=0)

        channel_df = pd.DataFrame({
            "Kanal": TARGET_CHANNELS,
            "Ortalama Dikkat Ağırlığı": mean_attn
        })

        channel_path = os.path.join(RESULT_DIR, "adaptive_channel_weights.csv")
        channel_df.to_csv(channel_path, index=False, encoding="utf-8-sig")

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(channel_df["Kanal"], channel_df["Ortalama Dikkat Ağırlığı"])
        ax.set_xlabel("PSG Kanalı")
        ax.set_ylabel("Ortalama Dikkat Ağırlığı")
        ax.set_title("Balanced Adaptive Fusion Kanal Ağırlıkları")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()

        fig_path = os.path.join(RESULT_DIR, "adaptive_channel_weights.png")
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.show()

        print("\nAdaptive channel weights:")
        print(channel_df)

    except Exception as e:
        print("Adaptive channel weights çıkarılamadı:", e)

    del adaptive_model, trained, hist, prob
    clear_memory()

# 10. SONUÇ TABLOSU VE GRAFİKLER

all_results_df = pd.DataFrame(all_results)

ordered_cols = [
    "Grup",
    "Model",
    "Accuracy",
    "Macro Precision",
    "Macro Recall",
    "Macro F1",
    "Weighted F1",
    "Kappa",
    "N1 Precision",
    "N1 Recall",
    "N1 F1",
    "N2 Precision",
    "N2 Recall",
    "N2 F1",
    "W F1",
    "N3 F1",
    "REM F1"
]

all_results_df = all_results_df[ordered_cols]

print("\n" + "=" * 80)
print("TÜM MODEL KARŞILAŞTIRMA SONUÇLARI")
print("=" * 80)
print(all_results_df.round(4))

results_path = os.path.join(RESULT_DIR, "tum_model_karsilastirma_sonuclari.csv")
all_results_df.to_csv(results_path, index=False, encoding="utf-8-sig")

print("Sonuç tablosu kaydedildi:", results_path)

plot_metric_bar(
    all_results_df,
    "Accuracy",
    "accuracy_tum_modeller.png",
    "Tüm Modellerin Accuracy Karşılaştırması"
)

plot_metric_bar(
    all_results_df,
    "Macro F1",
    "macro_f1_tum_modeller.png",
    "Tüm Modellerin Macro F1 Karşılaştırması"
)

plot_metric_bar(
    all_results_df,
    "Kappa",
    "kappa_tum_modeller.png",
    "Tüm Modellerin Kappa Karşılaştırması"
)

plot_metric_bar(
    all_results_df,
    "N1 F1",
    "n1_f1_tum_modeller.png",
    "Tüm Modellerin N1 F1 Karşılaştırması"
)

plot_metric_bar(
    all_results_df,
    "N2 F1",
    "n2_f1_tum_modeller.png",
    "Tüm Modellerin N2 F1 Karşılaştırması"
)

plot_all_metrics(all_results_df)

# 11. ÖZET


best_acc = all_results_df.sort_values("Accuracy", ascending=False).iloc[0]
best_macro = all_results_df.sort_values("Macro F1", ascending=False).iloc[0]
best_kappa = all_results_df.sort_values("Kappa", ascending=False).iloc[0]
best_n1 = all_results_df.sort_values("N1 F1", ascending=False).iloc[0]
best_n2 = all_results_df.sort_values("N2 F1", ascending=False).iloc[0]

summary_text = f"""
ÖZET SONUÇLAR
==================================================

Accuracy açısından en iyi model:
{best_acc['Model']} | Accuracy = {best_acc['Accuracy']:.4f}

Macro F1 açısından en iyi model:
{best_macro['Model']} | Macro F1 = {best_macro['Macro F1']:.4f}

Kappa açısından en iyi model:
{best_kappa['Model']} | Kappa = {best_kappa['Kappa']:.4f}

N1 F1 açısından en iyi model:
{best_n1['Model']} | N1 F1 = {best_n1['N1 F1']:.4f}

N2 F1 açısından en iyi model:
{best_n2['Model']} | N2 F1 = {best_n2['N2 F1']:.4f}

Ana önerilen model adayları:
1) Balanced_CNN_BiLSTM
2) Balanced_AdaptiveFusion

Not:
Bu kodda focal loss kullanılmamıştır.
Amaç N1 ve N3'ü artırırken N2'nin aşırı düşmesini engellemektir.
En iyi epoch seçimi val_accuracy yerine Macro F1 + N2 F1 + N1 F1 dengesine göre yapılmıştır.
"""

print(summary_text)

summary_path = os.path.join(RESULT_DIR, "ozet_sonuclar.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write(summary_text)

print("Özet kaydedildi:", summary_path)
print("Tüm çıktılar:", RESULT_DIR)

