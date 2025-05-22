import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ====== PATHS ======
MODEL_PATH = r"C:\Users\SHRAVINYA\Desktop\brain_tumor\brain_tumor_model.keras"
TEST_DIR = r"C:\Users\SHRAVINYA\Downloads\brain_tumor\Testing"

# ====== PARAMETERS ======
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ====== Load Model ======
print("[INFO] Loading model...")
model = load_model(MODEL_PATH)
print("[INFO] Model loaded successfully.")

# ====== Load Test Data ======
print("[INFO] Loading test data...")
test_datagen = ImageDataGenerator(rescale=1.0/255)
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# ====== Evaluate Model ======
print("[INFO] Evaluating model...")
loss, accuracy = model.evaluate(test_generator)
print(f"\n✅ Test Loss: {loss:.4f}")
print(f"✅ Test Accuracy: {accuracy:.4f}")

# ====== Predictions ======
print("[INFO] Making predictions...")
y_pred_probs = model.predict(test_generator)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# ====== Classification Report ======
print("\n📋 Classification Report:")
report = classification_report(y_true, y_pred, target_names=class_labels)
print(report)

# ====== Save Report to File ======
with open("evaluation_report.txt", "w") as f:
    f.write(report)
print("[INFO] Classification report saved as 'evaluation_report.txt'.")

# ====== Confusion Matrix ======
print("[INFO] Generating confusion matrix...")
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_labels, yticklabels=class_labels, cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()
print("[INFO] Confusion matrix saved as 'confusion_matrix.png'.")
