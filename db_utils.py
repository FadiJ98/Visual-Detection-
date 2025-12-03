# db_utils.py
from __future__ import annotations

from typing import Optional
import mysql.connector
import streamlit as st


def get_connection():
    """
    Open a new MySQL connection using Streamlit secrets.
    Requires .streamlit/secrets.toml with:

    [mysql]
    host = "localhost"
    user = "root"
    password = "your_password"
    database = "image_database"
    """
    cfg = st.secrets["mysql"]
    return mysql.connector.connect(
        host=cfg["host"],
        user=cfg["user"],
        password=cfg["password"],
        database=cfg["database"],
    )


def upsert_person(
    name: str,
    age: Optional[int] = None,
    gender: Optional[str] = None,
    emotion: Optional[str] = None,
    location: Optional[str] = None,
    lighting: Optional[str] = None,
) -> int:
    """
    Find an existing person by (name, age); if not found, insert a new one.
    Returns person_id.
    """
    conn = get_connection()
    try:
        cur = conn.cursor()

        # 1) Try to find existing person_id by name + age
        cur.execute(
            """
            SELECT person_id
            FROM persons
            WHERE name = %s
              AND (
                    (age IS NULL AND %s IS NULL)
                 OR (age = %s)
              )
            LIMIT 1
            """,
            (name, age, age),
        )
        row = cur.fetchone()
        if row:
            person_id = int(row[0])
            cur.close()
            return person_id

        # 2) Insert new person
        cur.execute(
            """
            INSERT INTO persons (name, age, gender, emotion, location, lighting)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (name, age, gender, emotion, location, lighting),
        )
        conn.commit()
        person_id = cur.lastrowid
        cur.close()
        return int(person_id)
    finally:
        conn.close()


def insert_image(
    person_id: int,
    file_path: str,
    pose: Optional[str] = None,
    tags: Optional[str] = None,
) -> int:
    """
    Insert a new image row for a person and return image_id.
    file_path should be how you store it on disk (e.g., 'faces_db/john_1.jpg').
    """
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO images (person_id, pose, file_path, tags)
            VALUES (%s, %s, %s, %s)
            """,
            (person_id, pose, file_path, tags),
        )
        conn.commit()
        image_id = cur.lastrowid
        cur.close()
        return int(image_id)
    finally:
        conn.close()


def insert_predicted_emotion(
    image_id: int,
    emotion: str,
    confidence: float,
) -> int:
    """
    Store a DeepFace emotion prediction for a given image.
    Emotion must match the ENUM:
    ('angry','disgust','fear','happy','sad','surprise','neutral')
    """
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO predicted_emotions (image_id, emotion, confidence)
            VALUES (%s, %s, %s)
            """,
            (image_id, emotion, confidence),
        )
        conn.commit()
        prediction_id = cur.lastrowid
        cur.close()
        return int(prediction_id)
    finally:
        conn.close()
