import sqlite3
from datetime import date
from typing import Dict


class Database:
    def __init__(self, db_path="data/predictions.db"):
        self.db_path = db_path
        self._init_db()

    def get_connection(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            image_name TEXT,
            predicted_class INTEGER,
            confidence REAL,
            user_id TEXT
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_stats (
            date TEXT PRIMARY KEY,
            total_predictions INTEGER,
            infarctus_count INTEGER,
            antecedents_count INTEGER,
            arythmie_count INTEGER,
            normal_count INTEGER,
            avg_confidence REAL,
            unique_users INTEGER
        )
        """)

        conn.commit()
        conn.close()

    # ===============================
    # SAVE PREDICTION
    # ===============================
    def save_prediction(self, prediction_data: Dict):
        conn = self.get_connection()
        cursor = conn.cursor()

        predicted_class = int(prediction_data.get("predicted_class", 0))
        confidence = float(prediction_data.get("confidence", 0.0))

        cursor.execute("""
        INSERT INTO predictions (timestamp, image_name, predicted_class, confidence, user_id)
        VALUES (?, ?, ?, ?, ?)
        """, (
            prediction_data.get("timestamp"),
            prediction_data.get("image_name"),
            predicted_class,
            confidence,
            prediction_data.get("user_id", "anonymous")
        ))

        conn.commit()
        conn.close()

        self._update_daily_stats({
            "predicted_class": predicted_class,
            "confidence": confidence
        })

    # ===============================
    # DAILY STATS (FIX CRITIQUE)
    # ===============================
    def _update_daily_stats(self, prediction_data: Dict):
        conn = self.get_connection()
        cursor = conn.cursor()

        today = date.today().isoformat()
        predicted_class = int(prediction_data.get("predicted_class", 0))
        confidence = float(prediction_data.get("confidence", 0.0))

        cursor.execute(
            "SELECT total_predictions, avg_confidence FROM daily_stats WHERE date = ?",
            (today,)
        )
        row = cursor.fetchone()

        class_map = {
            0: "infarctus_count",
            1: "antecedents_count",
            2: "arythmie_count",
            3: "normal_count"
        }

        if row:
            total, avg = row
            new_avg = ((avg * total) + confidence) / (total + 1)

            cursor.execute(f"""
            UPDATE daily_stats
            SET total_predictions = total_predictions + 1,
                avg_confidence = ?,
                {class_map[predicted_class]} = {class_map[predicted_class]} + 1
            WHERE date = ?
            """, (new_avg, today))
        else:
            counts = {v: 0 for v in class_map.values()}
            counts[class_map[predicted_class]] = 1

            cursor.execute("""
            INSERT INTO daily_stats (
                date, total_predictions,
                infarctus_count, antecedents_count,
                arythmie_count, normal_count,
                avg_confidence, unique_users
            ) VALUES (?, 1, ?, ?, ?, ?, ?, 1)
            """, (
                today,
                counts["infarctus_count"],
                counts["antecedents_count"],
                counts["arythmie_count"],
                counts["normal_count"],
                confidence
            ))

        conn.commit()
        conn.close()

    # ===============================
    # GLOBAL STATS
    # ===============================
    def get_global_statistics(self):
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM predictions")
        total_predictions = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT user_id) FROM predictions")
        unique_users = cursor.fetchone()[0]

        cursor.execute("SELECT AVG(confidence) FROM predictions")
        avg_confidence = cursor.fetchone()[0] or 0.0

        cursor.execute("""
        SELECT COUNT(*) FROM predictions
        WHERE date(timestamp) = date('now')
        """)
        today_predictions = cursor.fetchone()[0]

        conn.close()

        return {
            "total_predictions": total_predictions,
            "unique_users": unique_users,
            "avg_confidence": avg_confidence,
            "today_predictions": today_predictions
        }


def get_database():
    return Database()
