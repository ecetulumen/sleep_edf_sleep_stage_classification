# 2. NPZ VERİSİNİ YÜKLE

if not os.path.exists(CACHE_PATH):
    raise FileNotFoundError(
        f"Dosya bulunamadı: {CACHE_PATH}\n"
        "CACHE_PATH yolunu kontrol et."
    )

cache = np.load(CACHE_PATH, allow_pickle=True)

X_feat_seq = cache["X_feat_seq"].astype(np.float32)
X_feat_center = cache["X_feat_center"].astype(np.float32)
X_raw_center = cache["X_raw_center"].astype(np.float32)
y_seq = cache["y_seq"].astype(np.int64)
subject_seq = cache["subject_seq"]

if "feature_names" in cache:
    feature_names = cache["feature_names"]
else:
    feature_names = np.array([f"feature_{i}" for i in range(X_feat_center.shape[1])])

print("=" * 80)
print("Veri yüklendi.")
print("X_feat_seq:", X_feat_seq.shape)
print("X_feat_center:", X_feat_center.shape)
print("X_raw_center:", X_raw_center.shape)
print("y_seq:", y_seq.shape)
print("subject_seq:", subject_seq.shape)
print("=" * 80)

print("\nGenel sınıf dağılımı:")
print(pd.Series(y_seq).map(LABEL_TO_NAME).value_counts())

# 3. SUBJECT-WISE SPLIT

unique_subjects = np.unique(subject_seq)

train_subjects, temp_subjects = train_test_split(
    unique_subjects,
    test_size=0.30,
    random_state=RANDOM_STATE
)

val_subjects, test_subjects = train_test_split(
    temp_subjects,
    test_size=2/3,
    random_state=RANDOM_STATE
)

train_mask = np.isin(subject_seq, train_subjects)
val_mask = np.isin(subject_seq, val_subjects)
test_mask = np.isin(subject_seq, test_subjects)

Xfs_train = X_feat_seq[train_mask]
Xfs_val = X_feat_seq[val_mask]
Xfs_test = X_feat_seq[test_mask]

Xfc_train = X_feat_center[train_mask]
Xfc_val = X_feat_center[val_mask]
Xfc_test = X_feat_center[test_mask]

Xraw_train = X_raw_center[train_mask]
Xraw_val = X_raw_center[val_mask]
Xraw_test = X_raw_center[test_mask]

y_train = y_seq[train_mask]
y_val = y_seq[val_mask]
y_test = y_seq[test_mask]

print("\nSubject-wise split tamamlandı.")
print("Train:", Xraw_train.shape, Xfs_train.shape, y_train.shape)
print("Validation:", Xraw_val.shape, Xfs_val.shape, y_val.shape)
print("Test:", Xraw_test.shape, Xfs_test.shape, y_test.shape)

split_dist_df = pd.DataFrame({
    "Train": pd.Series(y_train).map(LABEL_TO_NAME).value_counts(),
    "Validation": pd.Series(y_val).map(LABEL_TO_NAME).value_counts(),
    "Test": pd.Series(y_test).map(LABEL_TO_NAME).value_counts()
}).fillna(0).astype(int)

print("\nTrain / Validation / Test sınıf dağılımı:")
print(split_dist_df)

split_dist_df.to_csv(
    os.path.join(RESULT_DIR, "train_validation_test_sinif_dagilimi.csv"),
    encoding="utf-8-sig"
)

# 4. FEATURE SCALING

n_features = Xfc_train.shape[1]

scaler_center = StandardScaler()
Xfc_train_scaled = scaler_center.fit_transform(Xfc_train)
Xfc_val_scaled = scaler_center.transform(Xfc_val)
Xfc_test_scaled = scaler_center.transform(Xfc_test)

scaler_seq = StandardScaler()

Xfs_train_2d = Xfs_train.reshape(-1, n_features)
Xfs_val_2d = Xfs_val.reshape(-1, n_features)
Xfs_test_2d = Xfs_test.reshape(-1, n_features)

Xfs_train_scaled = scaler_seq.fit_transform(Xfs_train_2d).reshape(Xfs_train.shape)
Xfs_val_scaled = scaler_seq.transform(Xfs_val_2d).reshape(Xfs_val.shape)
Xfs_test_scaled = scaler_seq.transform(Xfs_test_2d).reshape(Xfs_test.shape)

X_ml_trainval = np.vstack([Xfc_train_scaled, Xfc_val_scaled])
y_ml_trainval = np.concatenate([y_train, y_val])

sample_weight_ml = np.array([CLASS_WEIGHT[int(y)] for y in y_ml_trainval])

print("\nKullanılan class weight:")
for k, v in CLASS_WEIGHT.items():

