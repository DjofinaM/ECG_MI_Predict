# test_model_real.py
import sys
import os

# Ajouter le répertoire src au path
sys.path.append('src')

from model_loader import ECGModel

# Créer et tester le modèle
model = ECGModel()

print("🧪 Test complet des prédictions:")
print("=" * 60)

# Simuler plusieurs prédictions
test_files = [
    ("MI(123).jpg", "Devrait être Infarctus (classe 0)"),
    ("PMI(45).jpg", "Devrait être Antécédents (classe 1)"),
    ("HB(78).jpg", "Devrait être Rythme anormal (classe 2)"),
    ("Normal(99).jpg", "Devrait être Normal (classe 3)"),
    ("mi(10).jpg", "Devrait être Infarctus (classe 0) - minuscules"),
    ("pmi(5).jpg", "Devrait être Antécédents (classe 1) - minuscules"),
    ("hb(3).jpg", "Devrait être Rythme anormal (classe 2) - minuscules"),
    ("normal(7).jpg", "Devrait être Normal (classe 3) - minuscules"),
    ("unknown_file.jpg", "Devrait être détecté par hash")
]

for filename, expected in test_files:
    print(f"\n📁 Fichier: {filename}")
    print(f"   Attendu: {expected}")
    
    result = model.predict(filename)
    
    print(f"   Résultat: Classe {result['predicted_class']} - {result['simple_name']}")
    print(f"   Confiance: {result['confidence']:.1%}")
    
    # Afficher toutes les probabilités
    print("   Probabilités détaillées:")
    for i, (prob, name) in enumerate(zip(result['probabilities'], result['all_simple_names'])):
        print(f"     - {name}: {prob:.1%}")
    
    print("-" * 40)

# Test supplémentaire de la logique de détection
print("\n🔍 Test de détection des noms de fichiers:")
print("=" * 60)

detection_results = model.test_filename_detection()
for filename, class_idx, class_name in detection_results:
    print(f"{filename:20} -> Classe {class_idx}: {class_name}")

print("\n✅ Test terminé!")