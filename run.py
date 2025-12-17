# run.py - Point d'entrée principal
import sys
import os
import subprocess

# Ajouter le répertoire courant au path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Démarrage de l'application ECG Myocardite Detection...")
print(f"Répertoire de travail: {os.getcwd()}")
print(f"Python path: {sys.path[0]}")

# Importer et vérifier les dépendances
try:
    from src.database import get_database
    print("✅ Base de données importée")
    
    # Initialiser la base de données
    db = get_database()
    print("✅ Base de données initialisée")
    
    # Lancer Streamlit
    print("🚀 Lancement de l'application Streamlit...")
    subprocess.run(["streamlit", "run", "app/app.py"])
    
except Exception as e:
    print(f"❌ Erreur: {e}")
    print("\nInstallation des dépendances...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Réessayer
    from src.database import get_database
    db = get_database()
    subprocess.run(["streamlit", "run", "app/app.py"])