"""Command-line entry point; importing this module never starts an experiment."""

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import gc
import json
from pathlib import Path
import random
from uuid import uuid4

from .config import CLASS_WEIGHT, Config, MODEL_KEYS, N_CLASSES


def _versions():
    result = {}
    for name in ("numpy", "pandas", "matplotlib", "scikit-learn", "xgboost", "xgboost-cpu",
                 "tensorflow", "tensorflow-cpu", "keras"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            pass
    return result


def run_experiment(config):
    config.validate()
    if config.smoke_test:
        config = replace(config, epochs=1, batch_size=16, n_jobs=1, verbose=0)
    import numpy as np
    from .data_utils import load_dataset, make_demo_dataset, prepare_data, save_data_audit

    raw_data = (make_demo_dataset(config.random_state) if config.smoke_test else
                load_dataset(config.cache_path, trust_npz=config.trust_npz))
    data = prepare_data(raw_data, random_state=config.random_state)
    del raw_data
    if "xgb" in config.models and set(np.unique(data.y_trainval)) != set(range(N_CLASSES)):
        raise ValueError("XGBoost requires all five labels in train + validation; inspect the participant split.")
    random.seed(config.random_state)
    np.random.seed(config.random_state)
    dl_keys = set(config.models) - {"rf", "svm", "xgb"}
    tf = None
    if dl_keys:
        import tensorflow as tf
        from . import models as architectures
        from .training import predict_batched, train_and_evaluate_dl
        tf.keras.utils.set_random_seed(config.random_state)
        print("TensorFlow:", tf.__version__, "GPU:", tf.config.list_physical_devices("GPU"))
    import matplotlib
    if not config.show_plots:
        matplotlib.use("Agg")
    from .evaluation import Reporter, evaluate_predictions

    # Never overwrite earlier training outputs or accidentally restore an old checkpoint.
    prefix = "synthetic_smoke" if config.smoke_test else "experiment"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.result_dir) / f"{prefix}_{stamp}_{uuid4().hex[:6]}"
    reporter = Reporter(output_dir, show_plots=config.show_plots)
    print("Output:", output_dir.resolve())
    manifest = {"config": asdict(config), "versions": _versions(),
                "synthetic": config.smoke_test, "status": "running",
                "split_samples": {"train": len(data.train.y), "validation": len(data.val.y), "test": len(data.test.y)},
                "method_note": "ML fits train+validation; DL fits train and selects checkpoints on validation, as in source."}
    manifest_path = output_dir / "run_config.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    save_data_audit(data, output_dir)
    if config.smoke_test:
        (output_dir / "SYNTHETIC_DATA_ONLY.txt").write_text(
            "Synthetic software integration test. These scores are NOT Sleep-EDF research results.\n", encoding="utf-8")
    results = []
    try:
        for key in config.models:
            print(f"\nRunning: {key}")
            if key in ("rf", "svm", "xgb"):
                import joblib
                if key == "rf":
                    from sklearn.ensemble import RandomForestClassifier
                    name = "Random Forest"
                    model = RandomForestClassifier(
                        n_estimators=10 if config.smoke_test else 500, max_depth=20,
                        min_samples_leaf=2, random_state=config.random_state,
                        class_weight=CLASS_WEIGHT, n_jobs=config.n_jobs)
                elif key == "svm":
                    from sklearn.svm import SVC
                    name = "SVM"
                    model = SVC(kernel="rbf", C=3.0, gamma="scale", class_weight=CLASS_WEIGHT,
                                probability=False, random_state=config.random_state)
                else:
                    from xgboost import XGBClassifier
                    name = "XGBoost"
                    model = XGBClassifier(
                        objective="multi:softmax", num_class=N_CLASSES,
                        n_estimators=10 if config.smoke_test else 400,
                        max_depth=5, learning_rate=0.04, subsample=0.9, colsample_bytree=0.9,
                        reg_lambda=2.5, reg_alpha=0.3, eval_metric="mlogloss",
                        random_state=config.random_state, n_jobs=config.n_jobs)
                kwargs = {"sample_weight": data.sample_weight_ml} if key == "xgb" else {}
                model.fit(data.ml_trainval, data.y_trainval, **kwargs)
                predictions = model.predict(data.test.center)
                results.append(evaluate_predictions(data.test.y, predictions, name, "Makine Öğrenmesi"))
                reporter.predictions(data.test.y, predictions, name)
                joblib.dump(model, output_dir / f"{key}_model.joblib")
            else:
                raw_shape, seq_shape = data.train.raw.shape[1:], data.train.seq.shape[1:]
                feature_dim = data.train.center.shape[1]
                if key == "mlp":
                    model, name = architectures.build_mlp(feature_dim), "MLP"
                    choose = lambda split: split.center
                elif key == "cnn1d":
                    model, name = architectures.build_cnn_1d(raw_shape), "CNN_1D"
                    choose = lambda split: split.raw
                elif key == "bilstm":
                    model, name = architectures.build_bilstm(seq_shape), "BiLSTM_Attention"
                    choose = lambda split: split.seq
                elif key == "cnn_bilstm":
                    model = architectures.build_balanced_cnn_bilstm(raw_shape, seq_shape)
                    name = "Balanced_CNN_BiLSTM"
                    choose = lambda split: [split.raw, split.seq]
                else:
                    model = architectures.build_balanced_adaptive_fusion(raw_shape, seq_shape, feature_dim)
                    name = "Balanced_AdaptiveFusion"
                    choose = lambda split: [split.raw, split.seq, split.center]
                inputs = [choose(split) for split in (data.train, data.val, data.test)]
                result, model = train_and_evaluate_dl(
                    model, *inputs, y_train=data.train.y, y_val=data.val.y, y_test=data.test.y,
                    model_name=name, config=config, reporter=reporter)
                results.append(result)
                if key == "adaptive":
                    attention_model = tf.keras.Model(
                        inputs=model.inputs, outputs=model.get_layer("adaptive_channel_weights").output)
                    reporter.channel_weights(predict_batched(attention_model, inputs[2], config.batch_size))
                    del attention_model
                del inputs
            del model
            if tf is not None:
                tf.keras.backend.clear_session()
            gc.collect()
        frame = reporter.comparison(results, synthetic=config.smoke_test)
        print(frame.round(4).to_string(index=False))
        manifest["status"] = "completed"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return output_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description="Sleep-EDF classification: original eight-model experiment.")
    parser.add_argument("--cache-path", type=Path, default=Config.cache_path)
    parser.add_argument("--result-dir", type=Path, default=Config.result_dir)
    parser.add_argument("--models", nargs="+", choices=MODEL_KEYS, default=list(MODEL_KEYS))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--show-plots", action="store_true")
    parser.add_argument("--trust-npz", action="store_true", help="Allow pickle ONLY for your own trusted NPZ file.")
    parser.add_argument("--smoke-test", action="store_true", help="Synthetic data, 1 epoch and small trees; not research results.")
    parser.add_argument("--check-data", action="store_true", help="Check NPZ keys, shapes and participant split without training.")
    args = parser.parse_args(argv)
    config = Config(cache_path=args.cache_path, result_dir=args.result_dir, models=tuple(args.models),
                    epochs=args.epochs, batch_size=args.batch_size, random_state=args.seed,
                    n_jobs=args.n_jobs, show_plots=args.show_plots, trust_npz=args.trust_npz,
                    smoke_test=args.smoke_test)
    try:
        config.validate()
        if args.check_data:
            from .data_utils import load_dataset, make_demo_dataset, prepare_data
            dataset = (make_demo_dataset(args.seed) if args.smoke_test else
                       load_dataset(args.cache_path, trust_npz=args.trust_npz))
            prepared = prepare_data(dataset, random_state=args.seed)
            for name in ("train", "val", "test"):
                split = getattr(prepared, name)
                print(name, "raw:", split.raw.shape, "sequence:", split.seq.shape, "labels:", split.y.shape)
            print("Data checks passed. Training was NOT started.")
        else:
            output = run_experiment(config)
            print("Completed:", output.resolve())
    except (FileNotFoundError, ValueError, ImportError) as exc:
        parser.exit(2, f"Error: {exc}\nSee BASLANGIC_TR.md for setup and data instructions.\n")


if __name__ == "__main__":
    main()
