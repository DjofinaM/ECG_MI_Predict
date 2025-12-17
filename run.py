import streamlit as st
import sys
import os

# 1. Configurer les chemins pour que Python trouve vos dossiers 'src' et 'app'
# On récupère le chemin du dossier racine (ECG_MI_Predict)
root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)

# 2. Exécuter le code de votre application réelle
# Cela va lire et lancer votre fichier app/app.py
app_path = os.path.join(root_path, "app", "app.py")

try:
    with open(app_path, "rb") as source_file:
        code = compile(source_file.read(), app_path, "exec")
    exec(code, globals())
except Exception as e:
    st.error(f"Erreur lors du chargement de l'application : {e}")
