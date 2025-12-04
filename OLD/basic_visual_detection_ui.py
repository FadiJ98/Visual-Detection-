# basic_visual_detection_ui.py
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from deepface import DeepFace
from PIL import Image, ImageTk
from tkinter import Tk, Canvas, filedialog, messagebox
from tkinter import ttk

from recognition_deepface import RecognizerDeepFace


WINDOW_TITLE = "Visual Detection — DeepFace Only"
WINDOW_SIZE = "1920x1080"
BG_COLOR = "#1e1e1e"
CANVAS_BG = "#0f0f0f"
BOX_COLOR = (0, 255, 0)   # still here, but we now use per-face colors
TEXT_SCALE = 0.6
TEXT_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX

# ---------- COLOR PALETTE (BGR) ----------
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


class VisualDetectionApp:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry(WINDOW_SIZE)
        self.root.config(bg=BG_COLOR)
        self.root.minsize(800, 600)

        # State
        self.cv_bgr: Optional[np.ndarray] = None
        self.cv_annotated: Optional[np.ndarray] = None
        self.tk_img: Optional[ImageTk.PhotoImage] = None
        self.current_path: Optional[Path] = None
        self.last_emotions: List[str] = []
        self.last_names: List[str] = []
        self.last_ages: List[str] = []       # NEW
        self.last_genders: List[str] = []    # NEW

        # DeepFace recognizer (embeddings + DB)
        self.recognizer = RecognizerDeepFace(model_name="Facenet512")

        # UI
        self._build_header()
        self._build_toolbar()
        self._build_canvas()

        # Redraw on resize
        self.root.bind("<Configure>", lambda _: self._redraw())

        try:
            ttk.Style().theme_use("clam")
        except Exception:
            pass

    # ---------- UI BUILDERS ----------
    def _build_header(self) -> None:
        header = ttk.Frame(self.root, padding=12)
        header.pack(side="top", fill="x")

        ttk.Label(
            header,
            text="Face Detection + Emotion + Recognition (DeepFace)",
            font=("Segoe UI", 16, "bold"),
        ).pack(side="left")

    def _build_toolbar(self) -> None:
        toolbar = ttk.Frame(self.root, padding=10)
        toolbar.pack(side="top", fill="x")

        self.open_btn = ttk.Button(toolbar, text="Open Image", command=self.open_image)
        self.open_btn.pack(side="left", padx=5)

        self.detect_btn = ttk.Button(
            toolbar,
            text="Detect + Analyze",
            command=self.detect_faces,
            state="disabled",
        )
        self.detect_btn.pack(side="left", padx=5)

        self.show_boxes_btn = ttk.Button(
            toolbar,
            text="Show Boxes",
            command=lambda: self._redraw(annotated=True),
            state="disabled",
        )
        self.show_boxes_btn.pack(side="left", padx=5)

        self.save_btn = ttk.Button(
            toolbar,
            text="Save Annotated",
            command=self.save_annotated,
            state="disabled",
        )
        self.save_btn.pack(side="left", padx=5)

        # Clear Button
        self.clear_btn = ttk.Button(
            toolbar,
            text="Clear",
            command=self.clear_image,
        )
        self.clear_btn.pack(side="left", padx=5)

        self.status = ttk.Label(toolbar, text="Load an image to begin.")
        self.status.pack(side="right")

    def _build_canvas(self) -> None:
        self.canvas = Canvas(self.root, bg=CANVAS_BG, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=10, pady=10)

    # ---------- FILE OPS ----------
    def open_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        self.current_path = Path(path)
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", "Failed to load image.")
            return

        self.cv_bgr = img
        self.cv_annotated = None
        self.last_emotions.clear()
        self.last_names.clear()
        self.last_ages.clear()
        self.last_genders.clear()

        self.detect_btn.config(state="normal")
        self.save_btn.config(state="disabled")
        self.show_boxes_btn.config(state="disabled")

        self.status.config(text=f"Loaded: {self.current_path.name}")
        self._redraw()

    def save_annotated(self) -> None:
        if self.cv_annotated is None:
            messagebox.showinfo("Info", "Nothing to save.")
            return

        out = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile=(
                f"{self.current_path.stem}_annotated.png"
                if self.current_path
                else "annotated.png"
            ),
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("All files", "*.*"),
            ],
        )
        if not out:
            return

        cv2.imwrite(out, self.cv_annotated)
        self.status.config(text=f"Saved: {Path(out).name}")

    # ---------- CLEAR IMAGE ----------
    def clear_image(self) -> None:
        self.cv_bgr = None
        self.cv_annotated = None
        self.tk_img = None
        self.current_path = None
        self.last_emotions.clear()
        self.last_names.clear()
        self.last_ages.clear()
        self.last_genders.clear()

        self.detect_btn.config(state="disabled")
        self.save_btn.config(state="disabled")
        self.show_boxes_btn.config(state="disabled")

        self.status.config(text="Image cleared.")
        self._redraw()

    # ---------- DETECTION + EMOTION + AGE + GENDER + RECOGNITION ----------
    def detect_faces(self) -> None:
        if self.cv_bgr is None:
            return

        bgr = self.cv_bgr
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        try:
            result = DeepFace.analyze(
                img_path=rgb,
                actions=["emotion", "age", "gender"],
                enforce_detection=True,
                detector_backend="retinaface",
            )
        except Exception as e:
            messagebox.showerror("DeepFace error", str(e))
            return

        faces = result if isinstance(result, list) else [result]

        annotated = bgr.copy()
        self.last_emotions.clear()
        self.last_names.clear()
        self.last_ages.clear()
        self.last_genders.clear()

        h, w = bgr.shape[:2]

        for idx, r in enumerate(faces, start=1):
            region = r.get("region") or {}
            x = int(region.get("x", 0))
            y = int(region.get("y", 0))
            fw = int(region.get("w", 0))
            fh = int(region.get("h", 0))

            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w - 1, x + fw)
            y2 = min(h - 1, y + fh)
            if x2 <= x1 or y2 <= y1:
                continue

            face = bgr[y1:y2, x1:x2]

            # DeepFace attributes
            emotion = r.get("dominant_emotion", "unknown")
            age = r.get("age", None)
            gender = r.get("gender") or r.get("dominant_gender", "unknown")

            # Clean age
            if age is not None:
                try:
                    age = int(round(float(age)))
                except Exception:
                    pass

            self.last_emotions.append(str(emotion))
            self.last_ages.append(str(age) if age is not None else "?")
            self.last_genders.append(str(gender))

            # Recognition
            name, dist = self.recognizer.infer(face)
            self.last_names.append(name)

            # Choose color for this face (cycled)
            _, color_bgr = COLOR_PALETTE[(idx - 1) % len(COLOR_PALETTE)]

            # Draw ONLY a colored rectangle (no text on the image)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color_bgr, 3)

        self.cv_annotated = annotated
        self.save_btn.config(state="normal")
        self.show_boxes_btn.config(state="normal")

        if faces:
            self.status.config(
                text=(
                    f"Faces: {len(faces)} | "
                    f"Names: {', '.join(self.last_names)} | "
                    f"Ages: {', '.join(self.last_ages)} | "
                    f"Genders: {', '.join(self.last_genders)} | "
                    f"Emotions: {', '.join(self.last_emotions)}"
                )
            )
        else:
            self.status.config(text="No faces found.")

        self._redraw(annotated=True)

    # ---------- DRAW ----------
    def _redraw(self, annotated: bool = False) -> None:
        self.canvas.delete("all")

        img = self.cv_annotated if annotated and self.cv_annotated is not None else self.cv_bgr
        if img is None:
            self.canvas.create_text(
                self.canvas.winfo_width() // 2,
                self.canvas.winfo_height() // 2,
                text="Open an image to display",
                fill="#bbbbbb",
                font=("Segoe UI", 14),
            )
            return

        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        ih, iw = img.shape[:2]
        if cw <= 1 or ch <= 1:
            return

        scale = min(cw / iw, ch / ih)
        new_w, new_h = int(iw * scale), int(ih * scale)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb).resize((new_w, new_h), Image.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(pil_img)

        x = (cw - new_w) // 2
        y = (ch - new_h) // 2
        self.canvas.create_image(x, y, anchor="nw", image=self.tk_img)


def main() -> None:
    root = Tk()
    VisualDetectionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
