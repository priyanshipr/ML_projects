import os
import numpy as np
import tensorflow as tf
from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator

# ----------------------------
# 1. Paths
# ----------------------------
# Change this path if dataset is stored elsewhere
dataset_path = os.path.join("Animal Classification", "dataset")

# ----------------------------
# 2. Parameters
# ----------------------------
img_size = (128, 128)
batch_size = 32
epochs = 10

# ----------------------------
# 3. Data Generators
# ----------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_data = train_datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_data = train_datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# ----------------------------
# 4. Build Model (Transfer Learning)
# ----------------------------
base_model = tf.keras.applications.MobileNetV2(
    weights='imagenet', include_top=False, input_shape=(128,128,3)
)
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(train_data.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ----------------------------
# 5. Train Model
# ----------------------------
print("Training model...")
history = model.fit(train_data, validation_data=val_data, epochs=epochs)

# ----------------------------
# 6. Save Model
# ----------------------------
model.save("animal_classifier.h5")
print("Model saved as animal_classifier.h5")

# ----------------------------
# 7. Evaluate Model
# ----------------------------
loss, acc = model.evaluate(val_data)
print(f"Validation Accuracy: {acc*100:.2f}%")

# ----------------------------
# 8. Test Prediction
# ----------------------------
sample_img_path = val_data.filepaths[0]
img = image.load_img(sample_img_path, target_size=(128,128))
img_array = image.img_to_array(img)/255.0
img_array = np.expand_dims(img_array, axis=0)

pred = model.predict(img_array)
pred_class = np.argmax(pred)
class_labels = list(train_data.class_indices.keys())

print("Sample Image Path:", sample_img_path)
print("Predicted Class:", class_labels[pred_class])
