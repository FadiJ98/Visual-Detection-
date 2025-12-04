from __future__ import annotations
from typing import List
import streamlit as st

from recognition_deepface import RecognizerDeepFace


# ---------- COLOR PALETTE (name, BGR) ----------
COLOR_PALETTE: List[tuple[str, tuple[int, int, int]]] = [
    ("Blue",       (255,  80,  20)),
    ("Green",      ( 80, 220,  80)),
    ("Orange",     ( 40, 140, 255)),
    ("Yellow",     ( 40, 230, 255)),
    ("Purple",     (200,  80, 200)),
    ("Brown",      ( 40,  60, 140)),
    ("Gray",       (160, 160, 160)),
    ("Red",        ( 40,  40, 255)),
    ("Olive",      ( 60, 120,  60)),
    ("Maroon",     ( 40,  40, 140)),
    ("Violet",     (230, 130, 230)),
    ("Charcoal",   ( 60,  60,  60)),
    ("Magenta",    (230,  80, 230)),
    ("Bronze",     ( 60, 120, 200)),
    ("Cream",      (210, 220, 230)),
    ("Tan",        (140, 180, 220)),
    ("Teal",       (140, 200, 140)),
    ("Black",      (  0,   0,   0)),
    ("Mustard",    ( 60, 200, 220)),
    ("Navy Blue",  (180,  60,  40)),
    ("Coral",      (120, 160, 255)),
    ("Burgundy",   ( 40,  40, 110)),
    ("Lavender",   (220, 200, 250)),
    ("Mauve",      (200, 180, 220)),
    ("Peach",      (180, 200, 240)),
    ("Rust",       ( 60,  80, 160)),
    ("Gold",       ( 40, 200, 255)),
    ("Pink",       (220, 180, 250)),
    ("Silver",     (200, 200, 200)),
    ("Cyan",       (250, 220,  80)),
]

# ---------- FAKE NAMES BY GENDER (YOUR LISTS) ----------
FEMALE_NAMES = [
    "Sarah",
    "Emily",
    "Emma",
    "Sophia",
    "Olivia",
    "Patricia",
    "Stephanie",
    "Michelle",
    "Val",
    "Mirna",
    "Kaily",
    "Mandel",
    "Ammy",
]

MALE_NAMES = [
    "Mike",
    "Saher",
    "Jake",
    "Liam",
    "Noah",
    "Abdallah",
    "Hasan",
    "Mohamad",
    "Kathem",
    "Muneer",
    "Cal",
    "Andrew",
]


# ---------- CACHED RECOGNIZER (LAZY-LOADED) ----------
@st.cache_resource
def get_recognizer() -> RecognizerDeepFace:
    """
    Lazily create and cache the RecognizerDeepFace instance.

    This avoids loading Facenet512 + DB at import time, which is heavy
    in constrained environments like Streamlit Cloud.
    """
    return RecognizerDeepFace(model_name="Facenet512")
