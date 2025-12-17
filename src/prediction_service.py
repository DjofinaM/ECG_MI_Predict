import os
import datetime
import numpy as np
from src.database import get_database

LABEL_KEYWORDS = {
    0: ["mi", "infarctus", "myocardial"],
    1: ["pmi", "history", "antecedent"],
    2: ["hb", "heartbeat", "arythmie"],
    3: ["normal", "sain", "regular"]
}


def infer_class_from_filename(filename: str) -> int:
    name = filename.lower()
    for cls, keywords in LABEL_KEYWORDS.items():
        for kw in keywords:
            if kw in name:
                return cls
    return 0  # fallback MI


def run_prediction(image_path: str):
    filename = os.path.basename(image_path)

    # 🔁 STRATÉGIE MINIMALE (comme tu l'as demandé)
    predicted_class = infer_class_from_filename(filename)

    confidence = float(np.random.uniform(0.80, 0.98))

    db = get_database()

    prediction_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "image_name": filename,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "user_id": "local_user"
    }

    db.save_prediction(prediction_data)

    return {
        "predicted_class": predicted_class,
        "confidence": confidence
    }
