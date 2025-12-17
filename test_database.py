# test_database.py
import sqlite3
import os

# Créer le dossier data
os.makedirs("data", exist_ok=True)

# Connexion à la base
conn = sqlite3.connect("data/ecg_database.db")
cursor = conn.cursor()

# Créer les tables
cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_activity TIMESTAMP,
    total_predictions INTEGER DEFAULT 0,
    ip_address TEXT,
    user_agent TEXT
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    image_filename TEXT,
    image_size TEXT,
    image_hash TEXT,
    predicted_class INTEGER,
    class_name TEXT,
    simple_name TEXT,
    confidence REAL,
    probabilities TEXT,
    processing_time REAL,
    model_version TEXT,
    notes TEXT,
    FOREIGN KEY (user_id) REFERENCES users (id)
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS daily_stats (
    date DATE PRIMARY KEY,
    total_predictions INTEGER DEFAULT 0,
    normal_count INTEGER DEFAULT 0,
    infarctus_count INTEGER DEFAULT 0,
    arythmie_count INTEGER DEFAULT 0,
    antecedents_count INTEGER DEFAULT 0,
    avg_confidence REAL DEFAULT 0,
    unique_users INTEGER DEFAULT 0
)
''')

# Insérer des données de test
cursor.execute('''
INSERT OR IGNORE INTO users (session_id, last_activity, ip_address, user_agent, total_predictions)
VALUES ('test_session', datetime('now'), '127.0.0.1', 'Test Client', 5)
''')

cursor.execute('''
INSERT INTO predictions 
(user_id, image_filename, predicted_class, class_name, simple_name, confidence, model_version, timestamp)
VALUES 
(1, 'test1.jpg', 0, 'Infarctus du myocarde', 'Infarctus', 0.85, 'CNN_Test', datetime('now')),
(1, 'test2.jpg', 3, 'Normal', 'Normal', 0.92, 'CNN_Test', datetime('now')),
(1, 'test3.jpg', 2, 'Rythme cardiaque anormal', 'Rythme anormal', 0.78, 'CNN_Test', datetime('now'))
''')

conn.commit()
conn.close()

print("✅ Base de données créée avec des données de test!")
print("📊 Exécutez l'application et vérifiez les statistiques.")