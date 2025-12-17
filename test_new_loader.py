# test_demo_fixed.py
import sys
import os
sys.path.append('.')

from src.model_loader import get_model

print("🧪 TEST DE LA VERSION DÉMO CORRIGÉE")
print("=" * 50)

model = get_model()

# Test avec des noms de fichiers réalistes
test_files = [
    "data/raw/train/ECG Images of Myocardial Infarction Patients (240x12=2880)/MI(1).jpg",
    "data/raw/train/ECG Images of Patient that have History of MI (172x12=2064)/PMI(1).jpg",
    "data/raw/train/ECG Images of Patient that have abnormal heartbeat (233x12=2796)/HB(1).jpg",
    "data/raw/train/Normal Person ECG Images (284x12=3408)/Normal(1).jpg",
    "uploaded_ecg_123.jpg",  # Fichier sans motif
    "ecg_test_image.png"     # Autre fichier
]

for file_path in test_files:
    print(f"\n📁 Test: {os.path.basename(file_path)}")
    result = model.predict(file_path)
    
    if result["success"]:
        print(f"   🏥 Diagnostic: {result['simple_name']}")
        print(f"   📊 Confiance: {result['confidence']:.1%}")
    else:
        print(f"   ❌ Erreur: {result.get('error', 'Inconnue')}")