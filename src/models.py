import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

from .config import N_CLASSES

# 7. MODEL MİMARİLERİ

def temporal_attention(x, name_prefix="temporal"):
    score = layers.Dense(1, name=f"{name_prefix}_score")(x)
    weights = layers.Softmax(axis=1, name=f"{name_prefix}_weights")(score)
    weighted = layers.Multiply(name=f"{name_prefix}_multiply")([x, weights])

    context = layers.Lambda(
        lambda z: tf.reduce_sum(z, axis=1),
        name=f"{name_prefix}_context"
    )(weighted)

    return context


def conv_branch_raw_balanced(raw_inp, name_prefix="cnn"):
    x = layers.Conv1D(
        32, 9,
        padding="same",
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        name=f"{name_prefix}_conv1"
    )(raw_inp)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)
    x = layers.MaxPooling1D(2, name=f"{name_prefix}_pool1")(x)
    x = layers.Dropout(0.15, name=f"{name_prefix}_drop1")(x)

    x = layers.Conv1D(
        64, 7,
        padding="same",
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        name=f"{name_prefix}_conv2"
    )(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn2")(x)
    x = layers.MaxPooling1D(2, name=f"{name_prefix}_pool2")(x)
    x = layers.Dropout(0.20, name=f"{name_prefix}_drop2")(x)

    x = layers.Conv1D(
        96, 5,
        padding="same",
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        name=f"{name_prefix}_conv3"
    )(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn3")(x)
    x = layers.GlobalAveragePooling1D(name=f"{name_prefix}_gap")(x)

    x = layers.Dense(
        96,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        name=f"{name_prefix}_dense"
    )(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_dense_bn")(x)
    x = layers.Dropout(0.35, name=f"{name_prefix}_dense_drop")(x)

    return x


def build_mlp(input_dim):
    inp = layers.Input(shape=(input_dim,))

    x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.40)(x)

    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.30)(x)

    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.20)(x)

    out = layers.Dense(N_CLASSES, activation="softmax")(x)

    return models.Model(inp, out, name="MLP")


def build_cnn_1d(raw_shape):
    raw_inp = layers.Input(shape=raw_shape, name="raw_epoch_input")
    x = conv_branch_raw_balanced(raw_inp, name_prefix="cnn1d")
    out = layers.Dense(N_CLASSES, activation="softmax")(x)
    return models.Model(raw_inp, out, name="CNN_1D")


def build_bilstm(seq_shape):
    seq_inp = layers.Input(shape=seq_shape, name="feature_sequence_input")

    x = layers.Bidirectional(
        layers.LSTM(
            64,
            return_sequences=True,
            dropout=0.25,
            kernel_regularizer=regularizers.l2(1e-4)
        )
    )(seq_inp)

    x = layers.Bidirectional(
        layers.LSTM(
            32,
            return_sequences=True,
            dropout=0.25,
            kernel_regularizer=regularizers.l2(1e-4)
        )
    )(x)

    x = temporal_attention(x, name_prefix="bilstm_attention")

    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.30)(x)

    out = layers.Dense(N_CLASSES, activation="softmax")(x)

    return models.Model(seq_inp, out, name="BiLSTM_Attention")


def build_balanced_cnn_bilstm(raw_shape, seq_shape):
    raw_inp = layers.Input(shape=raw_shape, name="raw_epoch_input")
    seq_inp = layers.Input(shape=seq_shape, name="feature_sequence_input")

    x_raw = conv_branch_raw_balanced(raw_inp, name_prefix="balanced_cnn_bilstm_raw")

    x_seq = layers.Bidirectional(
        layers.LSTM(
            64,
            return_sequences=True,
            dropout=0.25,
            kernel_regularizer=regularizers.l2(1e-4)
        )
    )(seq_inp)

    x_seq = layers.Bidirectional(
        layers.LSTM(
            32,
            return_sequences=True,
            dropout=0.25,
            kernel_regularizer=regularizers.l2(1e-4)
        )
    )(x_seq)

    x_seq = temporal_attention(x_seq, name_prefix="balanced_cnn_bilstm_attention")

    x = layers.Concatenate(name="balanced_cnn_bilstm_fusion")([x_raw, x_seq])

    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.40)(x)

    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.25)(x)

    out = layers.Dense(N_CLASSES, activation="softmax")(x)

    return models.Model([raw_inp, seq_inp], out, name="Balanced_CNN_BiLSTM")


def build_balanced_adaptive_fusion(raw_shape, seq_shape, feature_dim):
    raw_inp = layers.Input(shape=raw_shape, name="raw_epoch_input")
    seq_inp = layers.Input(shape=seq_shape, name="feature_sequence_input")
    feat_inp = layers.Input(shape=(feature_dim,), name="center_feature_input")

    n_channels = raw_shape[-1]
    channel_embeddings = []

    for ch in range(n_channels):
        ch_signal = layers.Lambda(
            lambda z, c=ch: z[:, :, c:c+1],
            name=f"channel_{ch}_slice"
        )(raw_inp)

        x = layers.Conv1D(
            16, 9,
            padding="same",
            activation="relu",
            kernel_regularizer=regularizers.l2(1e-4)
        )(ch_signal)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Dropout(0.15)(x)

        x = layers.Conv1D(
            32, 7,
            padding="same",
            activation="relu",
            kernel_regularizer=regularizers.l2(1e-4)
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Dropout(0.20)(x)

        x = layers.Conv1D(
            64, 5,
            padding="same",
            activation="relu",
            kernel_regularizer=regularizers.l2(1e-4)
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling1D()(x)

        x = layers.Dense(
            64,
            activation="relu",
            kernel_regularizer=regularizers.l2(1e-4)
        )(x)
        x = layers.Dropout(0.25)(x)

        channel_embeddings.append(x)

    stacked = layers.Lambda(
        lambda tensors: tf.stack(tensors, axis=1),
        name="channel_embedding_stack"
    )(channel_embeddings)

    channel_scores = layers.Dense(1, name="adaptive_channel_score")(stacked)
    channel_weights = layers.Softmax(axis=1, name="adaptive_channel_weights")(channel_scores)

    weighted_channels = layers.Multiply(name="adaptive_channel_multiply")(
        [stacked, channel_weights]
    )

    fused_channel = layers.Lambda(
        lambda z: tf.reduce_sum(z, axis=1),
        name="adaptive_channel_fusion"
    )(weighted_channels)

    fused_channel = layers.BatchNormalization()(fused_channel)
    fused_channel = layers.Dropout(0.30)(fused_channel)

    x_seq = layers.Bidirectional(
        layers.LSTM(
            64,
            return_sequences=True,
            dropout=0.25,
            kernel_regularizer=regularizers.l2(1e-4)
        )
    )(seq_inp)

    x_seq = layers.Bidirectional(
        layers.LSTM(
            32,
            return_sequences=True,
            dropout=0.25,
            kernel_regularizer=regularizers.l2(1e-4)
        )
    )(x_seq)

    x_seq = temporal_attention(x_seq, name_prefix="adaptive_temporal_attention")

    x_seq = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x_seq)
    x_seq = layers.BatchNormalization()(x_seq)
    x_seq = layers.Dropout(0.30)(x_seq)

    x_feat = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(feat_inp)
    x_feat = layers.BatchNormalization()(x_feat)
    x_feat = layers.Dropout(0.30)(x_feat)

    x_feat = layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x_feat)
    x_feat = layers.Dropout(0.20)(x_feat)

    x = layers.Concatenate(name="balanced_adaptive_fusion_concat")(
        [fused_channel, x_seq, x_feat]
    )

    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.45)(x)

    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.25)(x)

    out = layers.Dense(N_CLASSES, activation="softmax")(x)

    return models.Model(
        [raw_inp, seq_inp, feat_inp],
        out,
        name="Balanced_AdaptiveFusion"
    )


