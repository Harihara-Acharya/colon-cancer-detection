#!/usr/bin/env python3
"""
Phase 2: EfficientNetB0 Training for Colon Cancer Classification
Requires: Run phase1_dataset.py first to prepare dataset/

M1 Mac Compatible | Frozen base model | Binary classification
"""

import os
import sys
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

try:
    import tensorflow as tf
    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.metrics import confusion_matrix, classification_report
    import seaborn as sns
    import math
except ImportError:
    print("Install: pip install tensorflow-macos tensorflow-metal")
    sys.exit(1)

# === CONFIGURATION ===
DATASET_DIR = Path('dataset')
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.0001
MAX_IMAGES_PER_CLASS = 200  # Match Phase 1

def verify_prerequisites():
    """Check dataset exists and TF ready"""
    print('=== Phase 2 Prerequisites ===')
    print(f'TensorFlow: {tf.__version__}')
    print(f'Dataset: {DATASET_DIR}')
    
    required = ['train', 'val', 'test']
    for split in required:
        if not (DATASET_DIR / split).exists():
            raise FileNotFoundError(f"Run phase1_dataset.py first: missing {DATASET_DIR / split}")
    
    gpus = tf.config.list_physical_devices('GPU')
    gpu_status = 'Yes (Metal)' if gpus else 'No (CPU)'
    print(f'GPU: {gpu_status}')
    print('Ready for training!')
    print()

def create_data_generators():
    """Create ImageDataGenerator for train/val/test"""
    print('=== Data Generators ===')
    
    # Train with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1
    )
    
    # Val/Test: only rescale
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        DATASET_DIR / 'train',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )
    
    val_generator = val_datagen.flow_from_directory(
        DATASET_DIR / 'val',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )
    
    test_generator = test_datagen.flow_from_directory(
        DATASET_DIR / 'test',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )
    
    print(f'Train batches: {train_generator.samples // BATCH_SIZE}')
    print(f'Val batches: {val_generator.samples // BATCH_SIZE}')
    print(f'Test batches: {test_generator.samples // BATCH_SIZE}')
    print()
    
    return train_generator, val_generator, test_generator

def build_model():
    """Build EfficientNetB0 with frozen base + classification head"""
    print('=== Building EfficientNetB0 ===')
    
    # Base model (frozen)
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(*IMG_SIZE, 3)
    )
    
    # Freeze base layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.summary()
    print(f'Trainable params: {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])}')
    print()
    
    return model

def train_model(model, train_gen, val_gen):
    """Compile, train, return history"""
    print('=== Training ===')
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    steps_per_epoch = math.ceil(train_gen.samples / BATCH_SIZE)
    validation_steps = math.ceil(val_gen.samples / BATCH_SIZE)

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=[early_stop],
        verbose=1
    )
    
    print('Training complete!')
    print()
    return history

def evaluate_and_plot(model, test_gen, history):
    """Test evaluation + accuracy/loss plots"""
    print('=== Evaluation ===')
    
    # Test evaluation
    test_loss, test_acc = model.evaluate(test_gen, verbose=0)
    print(f'Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)')

    # Predictions for confusion matrix
    y_pred = model.predict(test_gen)
    y_pred = (y_pred > 0.5).astype(int).flatten()
    y_true = test_gen.classes

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal','Cancer'],
                yticklabels=['Normal','Cancer'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    print()
    
    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Train')
    ax1.plot(history.history['val_accuracy'], label='Val')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Loss
    ax2.plot(history.history['loss'], label='Train')
    ax2.plot(history.history['val_loss'], label='Val')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print('Plots displayed.')

def save_model(model):
    """Save trained model"""
    model.save('model_phase1.h5')
    print("Model saved as 'model_phase1.h5'")
    print('Phase 2 COMPLETE! Ready for Phase 3 (attention).')

def main():
    verify_prerequisites()
    
    # Data pipeline
    train_gen, val_gen, test_gen = create_data_generators()
    
    # Model
    model = build_model()
    
    # Train
    history = train_model(model, train_gen, val_gen)
    
    # Evaluate
    evaluate_and_plot(model, test_gen, history)
    
    # Save
    save_model(model)

if __name__ == '__main__':
    main()

