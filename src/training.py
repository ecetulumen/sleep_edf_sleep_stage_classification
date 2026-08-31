import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score

from .config import CLASS_WEIGHT, N_CLASSES
from .evaluation import evaluate_predictions


def predict_batched(model, inputs, batch_size):
    """Inference without creating a new tf.data thread pool for every callback."""
    arrays = inputs if isinstance(inputs, (list, tuple)) else [inputs]
    count = len(arrays[0])
    if count == 0 or any(len(array) != count for array in arrays):
        raise ValueError("Prediction inputs must be nonempty and have matching sample counts.")
    output = []
    for start in range(0, count, batch_size):
        batch = [array[start:start + batch_size] for array in arrays]
        values = model(batch if isinstance(inputs, (list, tuple)) else batch[0], training=False)
        output.append(np.asarray(values))
    return np.concatenate(output)


def make_tf_dataset(inputs, labels, config, *, training=False):
    """Bounded-thread input pipeline; sample weights preserve the original class weights."""
    tensors = tuple(inputs) if isinstance(inputs, list) else inputs
    if training:
        weights = np.array([CLASS_WEIGHT[int(y)] for y in labels], dtype=np.float32)
        dataset = tf.data.Dataset.from_tensor_slices((tensors, labels, weights))
        dataset = dataset.shuffle(len(labels), seed=config.random_state, reshuffle_each_iteration=True)
    else:
        dataset = tf.data.Dataset.from_tensor_slices((tensors, labels))
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    return dataset.batch(config.batch_size).with_options(options).prefetch(1)


class BalancedF1Callback(tf.keras.callbacks.Callback):
    def __init__(self, val_data, y_val, model_name, config, output_dir):
        super().__init__()
        self.val_data, self.y_val = val_data, y_val
        self.config = config
        self.best_score = -np.inf
        self.wait = 0
        self.best_path = output_dir / f"{model_name}_best_balanced.weights.h5"
        self.saved_this_run = False

    def on_epoch_end(self, epoch, logs=None):
        probabilities = predict_batched(self.model, self.val_data, self.config.batch_size)
        pred = np.argmax(probabilities, axis=1)
        values = f1_score(self.y_val, pred, labels=list(range(N_CLASSES)), average=None, zero_division=0)
        macro_f1 = float(values.mean())
        _, n1, n2, n3, rem = values
        score = float(0.45 * macro_f1 + 0.25 * n2 + 0.15 * n1 + 0.10 * rem + 0.05 * n3)
        if logs is not None:
            logs["val_macro_f1"] = macro_f1
            logs["val_balanced_score"] = score
        if self.config.verbose:
            print(f"\nVal MacroF1={macro_f1:.4f} | N1F1={n1:.4f} | N2F1={n2:.4f} | "
                  f"N3F1={n3:.4f} | REMF1={rem:.4f} | BalancedScore={score:.4f}")
        if score > self.best_score:
            self.best_score, self.wait = score, 0
            self.model.save_weights(self.best_path)
            self.saved_this_run = True
        else:
            self.wait += 1
            if self.wait >= self.config.patience:
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.saved_this_run:
            self.model.load_weights(self.best_path)


def train_and_evaluate_dl(model, train_data, val_data, test_data, *, y_train, y_val,
                          y_test, model_name, config, reporter):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    if config.verbose:
        model.summary()
    balanced = BalancedF1Callback(val_data, y_val, model_name, config, reporter.output_dir)
    callbacks = [balanced, tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=config.verbose)]
    history = model.fit(
        make_tf_dataset(train_data, y_train, config, training=True),
        validation_data=make_tf_dataset(val_data, y_val, config),
        epochs=config.epochs, callbacks=callbacks, verbose=config.verbose,
    )
    probabilities = predict_batched(model, test_data, config.batch_size)
    predictions = np.argmax(probabilities, axis=1)
    result = evaluate_predictions(y_test, predictions, model_name, "Deep Learning")
    reporter.predictions(y_test, predictions, model_name, probabilities=probabilities)
    reporter.history(history, model_name)
    return result, model
