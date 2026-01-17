import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

# 1️⃣ Data augmentation for small dataset
augmenter = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    validation_split=0.2
)

# 2️⃣ Prepare training and validation generators
train_gen = augmenter.flow_from_directory(
    "dataset/",
    target_size=(224, 224),
    batch_size=16,
    class_mode="binary",
    subset="training"
)

val_gen = augmenter.flow_from_directory(
    "dataset/",
    target_size=(224, 224),
    batch_size=16,
    class_mode="binary",
    subset="validation"
)

# 3️⃣ Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# 4️⃣ Save the best model automatically
os.makedirs("app/model", exist_ok=True)
checkpoint = ModelCheckpoint(
    "app/model/pan_detector_model.h5",
    monitor="val_accuracy",
    save_best_only=True
)

# 5️⃣ Train model
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15,  # increase if needed
    callbacks=[checkpoint]
)
