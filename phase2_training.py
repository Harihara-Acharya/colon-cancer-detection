#!/usr/bin/env python3
"""
Phase 2: EfficientNetB0 Training for Colon Cancer Classification (FINAL FIXED)

✅ 2-Stage Training (Head → Fine-tune)
✅ EfficientNet preprocessing
✅ Proper unfreezing
✅ Clean imports & no syntax errors
"""

import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

try:
    import tensorflow as tf
    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras.applications.efficientnet import preprocess_input
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.metrics import AUC
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.metrics import confusion_matrix, classification_report
    import seaborn as sns
except ImportError:
    print("Install required packages first.")
    sys.exit(1)

# CONFIG
DATASET_DIR = Path('dataset')
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

STAGE1_EPOCHS = 5
STAGE2_EPOCHS = 20
STAGE1_LR = 1e-4
STAGE2_LR = 1e-5
UNFREEZE_LAYERS = 30


# =========================
# DATA GENERATORS
# =========================
def create_data_generators():
    print("=== Data Generators ===")

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    train_gen = train_datagen.flow_from_directory(
        DATASET_DIR / 'train',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=True,
        seed=42
    )

    val_gen = val_test_datagen.flow_from_directory(
        DATASET_DIR / 'val',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False,
        seed=42
    )

    test_gen = val_test_datagen.flow_from_directory(
        DATASET_DIR / 'test',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False,
        seed=42
    )

    return train_gen, val_gen, test_gen


# =========================
# MODEL
# =========================
def build_model():
    print("=== Building Model ===")

    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(*IMG_SIZE, 3)
    )

    # Freeze ALL layers initially
    for layer in base_model.layers:
        layer.trainable = False

    # Head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=output)

    return model, base_model


# =========================
# STAGE 1 TRAINING
# =========================
def train_stage1(model, train_gen, val_gen):
    print("=== Stage 1: Train Head ===")

    model.compile(
        optimizer=Adam(learning_rate=STAGE1_LR),
        loss='binary_crossentropy',
        metrics=['accuracy', AUC(name='auc')]
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        train_gen,
        epochs=STAGE1_EPOCHS,
        validation_data=val_gen,
        callbacks=[early_stop],
        class_weight={0: 1.0, 1: 1.0},
        verbose=1
    )

    return history


# =========================
# STAGE 2 TRAINING
# =========================
def train_stage2(model, base_model, train_gen, val_gen, history1):
    print("=== Stage 2: Fine-tuning ===")

    # Count BEFORE
    old_trainable = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])

    # Unfreeze last N layers
    for layer in base_model.layers[:-UNFREEZE_LAYERS]:
        layer.trainable = False
    for layer in base_model.layers[-UNFREEZE_LAYERS:]:
        layer.trainable = True

    # Count AFTER
    new_trainable = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])

    print(f"Trainable params: {new_trainable:,} (+{new_trainable - old_trainable:,})")

    model.compile(
        optimizer=Adam(learning_rate=STAGE2_LR),
        loss='binary_crossentropy',
        metrics=['accuracy', AUC(name='auc')]
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    history2 = model.fit(
        train_gen,
        epochs=STAGE2_EPOCHS,
        validation_data=val_gen,
        callbacks=[early_stop],
        class_weight={0: 1.0, 1: 1.0},
        verbose=1
    )

    # Merge histories
    for key in history1.history:
        history1.history[key] += history2.history[key]

    return history1


# =========================
# EVALUATION
# =========================
def evaluate(model, test_gen):
    print("=== Evaluation ===")

    loss, acc, auc = model.evaluate(test_gen, verbose=0)
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC: {auc:.4f}")

    y_pred_prob = model.predict(test_gen)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    y_true = test_gen.classes

    print("\nSample predictions:")
    print(y_pred[:20])

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.show()

    print(classification_report(y_true, y_pred))


# =========================
# MAIN
# =========================
def main():
    train_gen, val_gen, test_gen = create_data_generators()

    model, base_model = build_model()

    history1 = train_stage1(model, train_gen, val_gen)

    full_history = train_stage2(model, base_model, train_gen, val_gen, history1)

    evaluate(model, test_gen)

    model.save("final_model.h5")
    print("Model saved!")


if __name__ == "__main__":
    main()