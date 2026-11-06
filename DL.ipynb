# SMART WASTE CLASSIFICATION USING DEEP LEARNING

!pip -q install gradio scikit-learn seaborn

import os
import json
import zipfile
import urllib.request
import random
import shutil
import glob
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import gradio as gr

from PIL import Image
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import confusion_matrix, classification_report

# ----------------------------
# 1. ENVIRONMENT
# ----------------------------
SEED = 42
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
INITIAL_EPOCHS = 15
FINE_TUNE_EPOCHS = 5

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TensorFlow Version:", tf.__version__)
gpus = tf.config.list_physical_devices("GPU")
print("GPU Available:", gpus)

if not gpus:
    print(
        "\nWARNING: No GPU detected. Training will run on CPU and may be slow.\n"
        "In Colab, try Runtime > Change runtime type > Hardware accelerator > GPU.\n"
    )
else:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

# ----------------------------
# 2. DOWNLOAD AND EXTRACT DATASET
# ----------------------------
print("\nDownloading TrashNet dataset...")

dataset_url = "https://github.com/garythung/trashnet/archive/refs/heads/master.zip"
archive_path = "/content/trashnet.zip"
extract_root = "/content/trashnet-master"
expected_dataset_path = "/content/trashnet-master/data/dataset-resized"

if not os.path.exists(archive_path):
    urllib.request.urlretrieve(dataset_url, archive_path)

if not os.path.exists(extract_root):
    with zipfile.ZipFile(archive_path, "r") as zip_ref:
        zip_ref.extractall("/content")

nested_zip = "/content/trashnet-master/data/dataset-resized.zip"
if os.path.exists(nested_zip) and not os.path.isdir(expected_dataset_path):
    print("Extracting nested dataset archive...")
    with zipfile.ZipFile(nested_zip, "r") as zip_ref:
        zip_ref.extractall("/content/trashnet-master/data")

search_paths = glob.glob("/content/**/dataset-resized", recursive=True)
dataset_path = search_paths[0] if search_paths else expected_dataset_path

if not os.path.isdir(dataset_path):
    raise FileNotFoundError(
        f"TrashNet dataset folder was not found. Checked: {dataset_path}"
    )

categories = sorted(
    folder for folder in os.listdir(dataset_path)
    if os.path.isdir(os.path.join(dataset_path, folder))
)

if not categories:
    raise RuntimeError(f"No category folders found in {dataset_path}")

print("Dataset downloaded and extracted successfully!")
print("Dataset path:", dataset_path)
print("Categories:", categories)

# ----------------------------
# 3. DATASET DISTRIBUTION
# ----------------------------
category_counts = {}

for category in categories:
    folder = os.path.join(dataset_path, category)
    image_files = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    ]
    category_counts[category] = len(image_files)

print("\nDataset Distribution:")
print(pd.DataFrame(
    list(category_counts.items()),
    columns=["Category", "Image Count"]
).to_string(index=False))

# ----------------------------
# 4. VISUALIZE DATASET EXAMPLES
# ----------------------------
valid_categories = [
    category for category in categories if category_counts.get(category, 0) > 0
]

if valid_categories:
    cols = 3
    rows = int(np.ceil(len(valid_categories) / cols))
    plt.figure(figsize=(14, 4 * rows))

    for i, category in enumerate(valid_categories):
        folder = os.path.join(dataset_path, category)
        image_files = [
            f for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        ]
        try:
            with Image.open(os.path.join(folder, random.choice(image_files))) as img:
                img = img.convert("RGB")
                plt.subplot(rows, cols, i + 1)
                plt.imshow(img)
                plt.title(category.upper())
                plt.axis("off")
        except Exception as e:
            print(f"Skipping sample image for {category}: {e}")

    plt.suptitle("TrashNet Dataset Samples", fontsize=16)
    plt.tight_layout()
    plt.show()

plt.figure(figsize=(10, 5))
plt.bar(category_counts.keys(), category_counts.values())
plt.title("Waste Category Distribution")
plt.xlabel("Category")
plt.ylabel("Number of Images")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# ----------------------------
# 5. LOAD TRAINING AND VALIDATION DATA
# ----------------------------
print("\nLoading dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

class_names = train_ds.class_names
NUM_CLASSES = len(class_names)
print("Classes:", class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# ----------------------------
# 6. DATA AUGMENTATION
# ----------------------------
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1)
], name="data_augmentation")

# ----------------------------
# 7. BUILD MOBILENETV2 MODEL
# ----------------------------
print("\nLoading MobileNetV2...")

base_model = MobileNetV2(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="image")
x = data_augmentation(inputs)
# Preprocessing happens INSIDE the model. Do not preprocess again at prediction time.
x = layers.Rescaling(1.0 / 127.5, offset=-1, name="mobilenet_preprocessing")(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="class_probabilities")(x)

model = models.Model(inputs, outputs, name="smart_waste_classifier")
model.summary()

# ----------------------------
# 8. COMPILE MODEL
# ----------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ----------------------------
# 9. CHECKPOINTS AND CALLBACKS
# ----------------------------
model_dir = "/content/models"
os.makedirs(model_dir, exist_ok=True)

initial_checkpoint_path = os.path.join(model_dir, "best_initial.keras")
fine_checkpoint_path = os.path.join(model_dir, "best_finetuned.keras")
final_model_path = os.path.join(model_dir, "waste_classifier.keras")
class_names_path = os.path.join(model_dir, "class_names.json")

# Remove old checkpoints so this fresh run cannot accidentally select
# a checkpoint left over from an earlier interrupted run.
for old_checkpoint in [initial_checkpoint_path, fine_checkpoint_path, final_model_path]:
    if os.path.exists(old_checkpoint):
        os.remove(old_checkpoint)

initial_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    initial_checkpoint_path,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=1
)

# ----------------------------
# 10. INITIAL TRAINING
# ----------------------------
print("\nStarting initial training...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=INITIAL_EPOCHS,
    callbacks=[initial_checkpoint, early_stopping, reduce_lr],
    verbose=1
)

# Save the best initial model's weights before fine-tuning.
if os.path.exists(initial_checkpoint_path):
    initial_model = tf.keras.models.load_model(initial_checkpoint_path)
else:
    # Safety fallback in case no checkpoint was written.
    initial_model = model
    initial_model.save(initial_checkpoint_path)

# ----------------------------
# 11. FINE-TUNING
# ----------------------------
print("\nStarting fine-tuning...")

# Fine-tune only the last 30 base-model layers. Keep BatchNorm frozen.
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

fine_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    fine_checkpoint_path,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

fine_early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
    verbose=1
)

fine_reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=1e-7,
    verbose=1
)

fine_history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=[fine_checkpoint, fine_early_stopping, fine_reduce_lr],
    verbose=1
)

# ----------------------------
# 12. SELECT THE BEST AVAILABLE MODEL
# ----------------------------
print("\nSelecting the best available model...")

candidates = []

if os.path.exists(initial_checkpoint_path):
    candidates.append(("initial", tf.keras.models.load_model(initial_checkpoint_path)))

if os.path.exists(fine_checkpoint_path):
    candidates.append(("fine-tuned", tf.keras.models.load_model(fine_checkpoint_path)))

if not candidates:
    candidates.append(("current", model))

best_name = None
best_model = None
best_loss = float("inf")

for name, candidate in candidates:
    candidate_loss = candidate.evaluate(val_ds, verbose=0)[0]
    print(f"{name} model validation loss: {candidate_loss:.4f}")
    if candidate_loss < best_loss:
        best_loss = candidate_loss
        best_name = name
        best_model = candidate

print(f"Selected model: {best_name}")
best_model.save(final_model_path)

with open(class_names_path, "w", encoding="utf-8") as f:
    json.dump(class_names, f, indent=2)

# ----------------------------
# 13. ACCURACY AND LOSS GRAPHS
# ----------------------------
train_accuracy = history.history.get("accuracy", []) + fine_history.history.get("accuracy", [])
val_accuracy = history.history.get("val_accuracy", []) + fine_history.history.get("val_accuracy", [])
train_loss = history.history.get("loss", []) + fine_history.history.get("loss", [])
val_loss = history.history.get("val_loss", []) + fine_history.history.get("val_loss", [])

plt.figure(figsize=(10, 5))
plt.plot(train_accuracy, label="Training Accuracy")
plt.plot(val_accuracy, label="Validation Accuracy")
plt.title("Model Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("/content/accuracy.png", dpi=150)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(train_loss, label="Training Loss")
plt.plot(val_loss, label="Validation Loss")
plt.title("Model Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("/content/loss.png", dpi=150)
plt.show()

# ----------------------------
# 14. MODEL EVALUATION
# ----------------------------
print("\nEvaluating selected model...")
validation_loss, validation_accuracy = best_model.evaluate(val_ds, verbose=1)

print("\n========== MODEL RESULTS ==========")
print("Validation Accuracy:", round(validation_accuracy * 100, 2), "%")
print("Validation Loss:", round(validation_loss, 4))

# ----------------------------
# 15. CLASSIFICATION REPORT AND CONFUSION MATRIX
# ----------------------------
y_true = []
y_pred = []

for images, labels in val_ds:
    predictions = best_model.predict(images, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)
    y_true.extend(labels.numpy().tolist())
    y_pred.extend(predicted_labels.tolist())

print("\nClassification Report:\n")
print(classification_report(
    y_true,
    y_pred,
    labels=list(range(len(class_names))),
    target_names=class_names,
    zero_division=0
))

cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

plt.figure(figsize=(10, 7))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.title("Waste Classification Confusion Matrix")
plt.xlabel("Predicted Category")
plt.ylabel("Actual Category")
plt.tight_layout()
plt.savefig("/content/confusion_matrix.png", dpi=150)
plt.show()

# ----------------------------
# 16. RECYCLING GUIDANCE
# ----------------------------
recycling_guidance = {
    "cardboard": "Keep cardboard clean and dry. Flatten boxes before recycling.",
    "glass": "Separate glass and follow local glass recycling guidelines.",
    "metal": "Empty metal containers and follow local metal recycling rules.",
    "paper": "Keep paper dry and separate from food waste.",
    "plastic": "Check local recycling guidelines for the specific plastic type.",
    "trash": "This category needs further sorting. Check local disposal guidance."
}

def get_recommendation(category):
    return recycling_guidance.get(
        category,
        "Please check local waste management guidelines."
    )

# ----------------------------
# 17. PREDICTION FUNCTION
# ----------------------------
def predict_waste(image):
    if image is None:
        return None, 0.0, {}

    image = image.convert("RGB").resize(IMG_SIZE)
    image_array = np.asarray(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    # IMPORTANT: Do not call preprocess_input here.
    # The model applies preprocessing internally.
    predictions = best_model.predict(image_array, verbose=0)[0]

    predicted_index = int(np.argmax(predictions))
    predicted_class = class_names[predicted_index]
    confidence = float(predictions[predicted_index] * 100)

    scores = {
        class_names[i]: float(predictions[i])
        for i in range(len(class_names))
    }

    return predicted_class, confidence, scores

# ----------------------------
# 18. GRADIO APPLICATION
# ----------------------------
def classify_waste(image):
    if image is None:
        return {}, "Please upload an image."

    predicted_class, confidence, scores = predict_waste(image)

    result = (
        f"PREDICTED CATEGORY: {predicted_class.upper()}\n\n"
        f"Prediction Score: {confidence:.2f}%\n\n"
        f"RECYCLING GUIDANCE:\n{get_recommendation(predicted_class)}\n\n"
        "Note: AI predictions are not guaranteed disposal instructions."
    )
    return scores, result

# ----------------------------
# 19. EXPORT PROJECT FILES BEFORE LAUNCHING THE APP
# ----------------------------
output_dir = "/content/project_output"
os.makedirs(output_dir, exist_ok=True)

files_to_export = [
    final_model_path,
    class_names_path,
    "/content/accuracy.png",
    "/content/loss.png",
    "/content/confusion_matrix.png"
]

for file_path in files_to_export:
    if os.path.exists(file_path):
        shutil.copy2(file_path, output_dir)
    else:
        print("Warning: export file not found:", file_path)

zip_path = shutil.make_archive(
    "/content/Smart_Waste_Classification",
    "zip",
    output_dir
)

print("\n======================================")
print("MODEL TRAINING AND EXPORT COMPLETED!")
print("======================================")
print("Model: MobileNetV2")
print("Technique: Transfer Learning")
print("Classes:", class_names)
print("Validation Accuracy:", round(validation_accuracy * 100, 2), "%")
print("Model location:", final_model_path)
print("Project ZIP:", zip_path)
print("======================================")

# ----------------------------
# 20. LAUNCH GRADIO APPLICATION LAST
# ----------------------------
print("\nLaunching Smart Waste Classification Application...")

demo = gr.Interface(
    fn=classify_waste,
    inputs=gr.Image(type="pil", label="Upload Waste Image"),
    outputs=[
        gr.Label(num_top_classes=3, label="Classification Results"),
        gr.Textbox(label="Recycling Guidance", lines=6)
    ],
    title="Smart Waste Classification System",
    description=(
        "AI-powered waste classification using MobileNetV2 and Transfer Learning. "
        "Upload an image to classify it into Cardboard, Glass, Metal, Paper, "
        "Plastic, or Trash."
    ),
    theme=gr.themes.Soft()
)

demo.launch(share=True, debug=False)
