import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)
from tensorflow.keras.optimizers import Adam

# =====================================================
# Configuration
# =====================================================

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 16

EPOCHS_STAGE1 = 5

EPOCHS_STAGE2 = 15

TRAIN_DIR = "dataset_split/train"

VALID_DIR = "dataset_split/valid"

TEST_DIR = "dataset_split/test"

MODEL_PATH = "models/package_classifier.keras"

os.makedirs("models", exist_ok=True)

# =====================================================
# Data Generators
# =====================================================

train_datagen = ImageDataGenerator(

    preprocessing_function=preprocess_input,

    rotation_range=20,

    zoom_range=0.20,

    width_shift_range=0.20,

    height_shift_range=0.20,

    horizontal_flip=True,

    fill_mode="nearest"

)

valid_datagen = ImageDataGenerator(

    preprocessing_function=preprocess_input

)

test_datagen = ImageDataGenerator(

    preprocessing_function=preprocess_input

)

train_generator = train_datagen.flow_from_directory(

    TRAIN_DIR,

    target_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="binary",

    shuffle=True

)

valid_generator = valid_datagen.flow_from_directory(

    VALID_DIR,

    target_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="binary",

    shuffle=False

)

test_generator = test_datagen.flow_from_directory(

    TEST_DIR,

    target_size=IMAGE_SIZE,

    batch_size=1,

    class_mode="binary",

    shuffle=False

)

print("\nClass Mapping")

print(train_generator.class_indices)

# =====================================================
# Load EfficientNetB0
# =====================================================

base_model = EfficientNetB0(

    include_top=False,

    weights="imagenet",

    input_shape=(224,224,3)

)

base_model.trainable = False

# =====================================================
# Classification Head
# =====================================================

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dropout(0.40)(x)

output = Dense(

    1,

    activation="sigmoid"

)(x)

model = Model(

    inputs=base_model.input,

    outputs=output

)

# =====================================================
# Compile
# =====================================================

model.compile(

    optimizer=Adam(learning_rate=0.001),

    loss="binary_crossentropy",

    metrics=["accuracy"]

)

model.summary()

# =====================================================
# Callbacks
# =====================================================

checkpoint = ModelCheckpoint(

    MODEL_PATH,

    monitor="val_accuracy",

    save_best_only=True,

    verbose=1

)

early_stop = EarlyStopping(

    monitor="val_loss",

    patience=5,

    restore_best_weights=True

)

reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.2,

    patience=2,

    verbose=1,

    min_lr=1e-6

)

# =====================================================
# Stage 1 Training
# =====================================================

print("\n========================")

print("Stage 1 Training")

print("========================\n")

history1 = model.fit(

    train_generator,

    validation_data=valid_generator,

    epochs=EPOCHS_STAGE1,

    callbacks=[

        checkpoint,

        early_stop,

        reduce_lr

    ]

)