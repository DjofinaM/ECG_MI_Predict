"""
Base de données SQLite pour l'application ECG.
Stocke les utilisateurs, prédictions et statistiques.
"""

import sqlite3
import json
import os
from datetime import datetime, date
from typing import List, Dict, Optional
import streamlit as st
import pandas as pd
import numpy as np

class ECGDatabase:
    """Gestionnaire de base de données SQLite pour l'application ECG."""
    
    def __init__(self, db_path: str = "data/ecg_database.db"):
        """Initialise la base de données."""
        # Créer le dossier data s'il n'existe pas
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.db_path = db_path
        self.init_database()
    
    def get_connection(self):
        """Retourne une connexion à la base de données."""
        return sqlite3.connect(self.db_path)
    
    def init_database(self):
        """Initialise les tables de la base de données."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Table des utilisateurs
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
        
        # Table des prédictions
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
        
        # Table des statistiques journalières
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
        
        # Index pour accélérer les recherches
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_predictions_user ON predictions(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_predictions_class ON predictions(predicted_class)')
        
        conn.commit()
        conn.close()
    
    def create_or_get_user(self, session_id: str, ip_address: str = None, user_agent: str = None) -> int:
        """Crée ou récupère un utilisateur basé sur la session Streamlit."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Vérifier si l'utilisateur existe
        cursor.execute('SELECT id, total_predictions FROM users WHERE session_id = ?', (session_id,))
        result = cursor.fetchone()
        
        if result:
            user_id, pred_count = result
            # Mettre à jour le timestamp et les infos
            cursor.execute('''
                UPDATE users 
                SET last_activity = ?, ip_address = ?, user_agent = ?
                WHERE id = ?
            ''', (datetime.now().isoformat(), ip_address, user_agent, user_id))
        else:
            # Créer un nouvel utilisateur
            cursor.execute('''
                INSERT INTO users (session_id, last_activity, ip_address, user_agent) 
                VALUES (?, ?, ?, ?)
            ''', (session_id, datetime.now().isoformat(), ip_address, user_agent))
            user_id = cursor.lastrowid
            pred_count = 0
        
        conn.commit()
        conn.close()
        
        return user_id, pred_count
    
    def save_prediction(self, user_id: int, prediction_data: Dict) -> int:
        """Sauvegarde une prédiction dans la base de données."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Calculer un hash simple de l'image (pour éviter les doublons)
        import hashlib
        image_hash = hashlib.md5(prediction_data.get('image_filename', '').encode()).hexdigest()[:16]
        
        # Insérer la prédiction
        cursor.execute('''
            INSERT INTO predictions 
            (user_id, image_filename, image_size, image_hash, predicted_class, 
             class_name, simple_name, confidence, probabilities, 
             processing_time, model_version, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_id,
            prediction_data.get('image_filename', 'unknown'),
            prediction_data.get('image_size', '0x0'),
            image_hash,
            prediction_data.get('predicted_class', 0),
            prediction_data.get('class_name', 'Unknown'),
            prediction_data.get('simple_name', 'Unknown'),
            prediction_data.get('confidence', 0.0),
            json.dumps(prediction_data.get('probabilities', []), indent=2),
            prediction_data.get('processing_time', 0.0),
            prediction_data.get('model_version', 'CNN'),
            prediction_data.get('notes', '')
        ))
        
        prediction_id = cursor.lastrowid
        
        # Mettre à jour le compteur de prédictions de l'utilisateur
        cursor.execute(
            'UPDATE users SET total_predictions = total_predictions + 1 WHERE id = ?',
            (user_id,)
        )
        
        # Mettre à jour les statistiques journalières
        self._update_daily_stats(prediction_data)
        
        conn.commit()
        conn.close()
        
        return prediction_id
    
    def _update_daily_stats(self, prediction_data: Dict):
        """Met à jour les statistiques journalières."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        today = date.today().isoformat()
        
        # Vérifier si des statistiques existent pour aujourd'hui
        cursor.execute('SELECT * FROM daily_stats WHERE date = ?', (today,))
        result = cursor.fetchone()
        
        predicted_class = prediction_data.get('predicted_class', 0)
        confidence = prediction_data.get('confidence', 0.0)
        
        if result:
            # Mettre à jour les statistiques existantes
            cursor.execute('''
                UPDATE daily_stats 
                SET total_predictions = total_predictions + 1,
                    avg_confidence = ((avg_confidence * total_predictions) + ?) / (total_predictions + 1)
                WHERE date = ?
            ''', (confidence, today))
            
            # Mettre à jour le compteur par classe
            class_columns = {
                0: 'infarctus_count',
                1: 'antecedents_count', 
                2: 'arythmie_count',
                3: 'normal_count'
            }
            
            if predicted_class in class_columns:
                column = class_columns[predicted_class]
                cursor.execute(f'''
                    UPDATE daily_stats 
                    SET {column} = {column} + 1 
                    WHERE date = ?
                ''', (today,))
        else:
            # Créer de nouvelles statistiques
            counts = [0, 0, 0, 0]
            if 0 <= predicted_class <= 3:
                counts[predicted_class] = 1
            
            cursor.execute('''
                INSERT INTO daily_stats 
                (date, total_predictions, normal_count, infarctus_count, 
                 arythmie_count, antecedents_count, avg_confidence, unique_users)
                VALUES (?, 1, ?, ?, ?, ?, ?, 1)
            ''', (today, counts[3], counts[0], counts[2], counts[1], confidence))
        
        conn.commit()
        conn.close()
    
    def get_user_predictions(self, user_id: int, limit: int = 20) -> List[Dict]:
        """Récupère les prédictions d'un utilisateur."""
        conn = self.get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                id, timestamp, image_filename, predicted_class,
                class_name, simple_name, confidence, probabilities,
                processing_time, model_version
            FROM predictions 
            WHERE user_id = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (user_id, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        # Convertir en liste de dictionnaires
        predictions = []
        for row in rows:
            pred = dict(row)
            try:
                pred['probabilities'] = json.loads(pred['probabilities'])
            except:
                pred['probabilities'] = []
            predictions.append(pred)
        
        return predictions
    
    def get_user_statistics(self, user_id: int) -> Dict:
        """Récupère les statistiques d'un utilisateur."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Nombre total de prédictions
        cursor.execute('SELECT total_predictions FROM users WHERE id = ?', (user_id,))
        result = cursor.fetchone()
        total_predictions = result[0] if result else 0
        
        # Distribution des classes
        cursor.execute('''
            SELECT predicted_class, COUNT(*) as count
            FROM predictions
            WHERE user_id = ?
            GROUP BY predicted_class
            ORDER BY predicted_class
        ''', (user_id,))
        
        class_dist = cursor.fetchall()
        
        # Confidence moyenne
        cursor.execute('SELECT AVG(confidence) FROM predictions WHERE user_id = ?', (user_id,))
        avg_confidence = cursor.fetchone()[0] or 0
        
        # Dernière activité
        cursor.execute('SELECT last_activity FROM users WHERE id = ?', (user_id,))
        last_activity = cursor.fetchone()
        last_activity = last_activity[0] if last_activity else None
        
        conn.close()
        
        # Organiser la distribution des classes
        class_distribution = {0: 0, 1: 0, 2: 0, 3: 0}
        for class_id, count in class_dist:
            if class_id in class_distribution:
                class_distribution[class_id] = count
        
        return {
            'total_predictions': total_predictions,
            'class_distribution': class_distribution,
            'avg_confidence': round(avg_confidence, 3),
            'last_activity': last_activity
        }
    
    def get_global_statistics(self) -> Dict:
        """Récupère les statistiques globales."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Totaux généraux
        cursor.execute('SELECT COUNT(*) FROM predictions')
        total_predictions = cursor.fetchone()[0] or 0
        
        cursor.execute('SELECT COUNT(DISTINCT user_id) FROM predictions')
        total_users = cursor.fetchone()[0] or 0
        
        cursor.execute('SELECT AVG(confidence) FROM predictions')
        avg_confidence = cursor.fetchone()[0] or 0
        
        cursor.execute('SELECT COUNT(DISTINCT date(timestamp)) FROM predictions')
        total_days = cursor.fetchone()[0] or 0
        
        # Distribution par classe
        cursor.execute('''
            SELECT predicted_class, COUNT(*) as count 
            FROM predictions 
            GROUP BY predicted_class 
            ORDER BY predicted_class
        ''')
        class_dist_raw = cursor.fetchall()
        
        class_names = ['Infarctus', 'Antécédents', 'Rythme anormal', 'Normal']
        class_distribution = {}
        for class_id, count in class_dist_raw:
            if 0 <= class_id <= 3:
                class_distribution[class_names[class_id]] = count
        
        # Statistiques des 7 derniers jours
        cursor.execute('''
            SELECT date(timestamp) as day, COUNT(*) as count
            FROM predictions
            WHERE timestamp >= date('now', '-7 days')
            GROUP BY date(timestamp)
            ORDER BY day
        ''')
        weekly_stats = cursor.fetchall()
        
        # Modèles utilisés
        cursor.execute('''
            SELECT model_version, COUNT(*) as count
            FROM predictions
            GROUP BY model_version
        ''')
        model_stats = dict(cursor.fetchall())
        
        conn.close()
        
        return {
            'total_predictions': total_predictions,
            'total_users': total_users,
            'total_days': total_days,
            'avg_confidence': round(avg_confidence, 3),
            'avg_daily_predictions': round(total_predictions / max(total_days, 1), 1),
            'class_distribution': class_distribution,
            'weekly_stats': weekly_stats,
            'model_stats': model_stats
        }
    
    def export_user_data(self, user_id: int, format: str = 'json') -> str:
        """Exporte les données d'un utilisateur."""
        predictions = self.get_user_predictions(user_id, limit=1000)
        
        if format == 'json':
            import json
            filename = f"data/export/user_{user_id}_data.json"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(predictions, f, indent=2, ensure_ascii=False)
            return filename
        elif format == 'csv':
            filename = f"data/export/user_{user_id}_data.csv"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            # Convertir en DataFrame
            df_data = []
            for pred in predictions:
                row = {
                    'id': pred['id'],
                    'timestamp': pred['timestamp'],
                    'image_filename': pred['image_filename'],
                    'predicted_class': pred['predicted_class'],
                    'class_name': pred['class_name'],
                    'confidence': pred['confidence'],
                    'processing_time': pred['processing_time'],
                    'model_version': pred['model_version']
                }
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            df.to_csv(filename, index=False, encoding='utf-8')
            return filename
        
        return ""
    
    def export_all_data(self, format: str = 'csv') -> str:
        """Exporte toutes les données."""
        conn = self.get_connection()
        
        if format == 'csv':
            # Exporter les prédictions
            query = '''
                SELECT 
                    p.id, p.timestamp, u.session_id, 
                    p.image_filename, p.predicted_class, p.class_name,
                    p.simple_name, p.confidence, p.processing_time, p.model_version
                FROM predictions p
                JOIN users u ON p.user_id = u.id
                ORDER BY p.timestamp
            '''
            df = pd.read_sql_query(query, conn)
            
            filename = "data/export/all_predictions.csv"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            df.to_csv(filename, index=False, encoding='utf-8')
            
            # Exporter les statistiques journalières
            query_stats = "SELECT * FROM daily_stats ORDER BY date"
            df_stats = pd.read_sql_query(query_stats, conn)
            stats_filename = "data/export/daily_stats.csv"
            df_stats.to_csv(stats_filename, index=False, encoding='utf-8')
            
            conn.close()
            return filename, stats_filename
        
        conn.close()
        return ""
    
    def backup_database(self):
        """Crée une sauvegarde de la base de données."""
        backup_path = f"data/backup/ecg_db_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        
        import shutil
        shutil.copy2(self.db_path, backup_path)
        
        return backup_path

# Instance globale de la base de données
_db_instance = None

def get_database():
    """Retourne l'instance unique de la base de données."""
    global _db_instance
    if _db_instance is None:
        _db_instance = ECGDatabase()
    return _db_instance