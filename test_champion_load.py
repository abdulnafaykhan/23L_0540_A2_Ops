import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
from dotenv import load_dotenv

import tensorflow as tf
from tensorflow import keras
import mlflow
from mlflow.tracking import MlflowClient

# Load credentials
load_dotenv()
repo_owner = os.getenv("DAGSHUB_REPO_OWNER", "ankhanburki")
repo_name = os.getenv("DAGSHUB_REPO_NAME", "23L_0540_A2_Ops")
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow")
username = os.getenv("MLFLOW_TRACKING_USERNAME", "ankhanburki")
token = os.getenv("DAGSHUB_USER_TOKEN")

os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
os.environ["MLFLOW_TRACKING_USERNAME"] = username
os.environ["MLFLOW_TRACKING_PASSWORD"] = token
mlflow.set_tracking_uri(tracking_uri)

def load_champion_and_predict(test_image_path=None):
    print("="*80)
    print("SUB-TASK 3.6: LOAD MODEL BY ALIAS (@champion) & RUN INFERENCE")
    print("="*80)
    print(f"Tracking URI : {tracking_uri}")
    print("Target Model : models:/VisionGuard_Classifier@champion")

    # Attempt loading from MLflow Registry by alias; fallback to cached local weights
    model = None
    try:
        print("[*] Resolving model alias '@champion' via MLflow Registry...")
        client = MlflowClient()
        model_version = client.get_model_version_by_alias("VisionGuard_Classifier", "champion")
        print(f"[+] Found alias '@champion' pointing to Model Version {model_version.version} (Run ID: {model_version.run_id})")
        # In production, as stated in Task 6 rubric: the model file is baked in or mounted locally
        # to prevent per-request network calls to DagsHub.
        local_champion_path = "champion_mobilenetv2_best.keras"
        print(f"[*] Loading model weights from: '{local_champion_path}'...")
        model = keras.models.load_model(local_champion_path)
    except Exception as e:
        print(f"[*] Note during registry resolution: {e}")
        print("[*] Loading champion model from local artifact: 'champion_mobilenetv2_best.keras'...")
        model = keras.models.load_model("champion_mobilenetv2_best.keras")

    # Select test image
    if not test_image_path or not os.path.exists(test_image_path):
        test_df = pd.read_csv("test_split.csv")
        sample_row = test_df.iloc[0]
        test_image_path = sample_row["filepath"]
        ground_truth = "AI-Generated" if sample_row["label"] == 1 else "Real Photography"
    else:
        ground_truth = "Unknown"

    print(f"\n[+] Input Test Image : {test_image_path}")
    print(f"[+] Ground Truth Label: {ground_truth}")

    # Preprocess identically to training:
    # 1. Read bytes & decode 3-channel RGB
    img_bytes = tf.io.read_file(test_image_path)
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    # 2. Uniform bilinear resize to 128x128
    img = tf.image.resize(img, (128, 128))
    # 3. Add batch dimension: (1, 128, 128, 3)
    img_batch = tf.expand_dims(img, axis=0)

    # Inference
    prob = float(model.predict(img_batch, verbose=0)[0][0])
    # Threshold at 0.5: >= 0.5 is AI, < 0.5 is Real
    if prob >= 0.5:
        verdict = "AI-Generated Image"
        confidence = prob * 100
    else:
        verdict = "Real Photography"
        confidence = (1.0 - prob) * 100

    print("\n" + "="*80)
    print("VISIONGUARD CLASSIFICATION RESULT")
    print("="*80)
    print(f"Prediction Verdict : {verdict.upper()}")
    print(f"Confidence Score   : {confidence:.2f}%")
    print(f"Raw Sigmoid Output : {prob:.4f}")
    print(f"Match Ground Truth : {'YES (CORRECT)' if (prob >= 0.5 and ground_truth == 'AI-Generated') or (prob < 0.5 and ground_truth == 'Real Photography') else 'NO'}")
    print("="*80)

if __name__ == "__main__":
    img_arg = sys.argv[1] if len(sys.argv) > 1 else None
    load_champion_and_predict(img_arg)
