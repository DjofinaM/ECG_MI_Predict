from src.model_loader import get_model
from src.database import get_database

def run_prediction(image_path, user_id="demo_user"):
    model = get_model()
    db = get_database()

    predicted_class, confidence = model.predict(image_path)

    db.save_prediction(
        user_id=user_id,
        predicted_class=predicted_class,
        confidence=confidence
    )

    return {
        "class": predicted_class,
        "confidence": confidence
    }
