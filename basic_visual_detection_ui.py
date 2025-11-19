import cv2
import numpy as np
from pathlib import Path
from tkinter import Tk, Canvas, filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk

from detection_mediapipe import FaceDetectorMP


class VisualDetectionApp:
    def __init__(self, root: Tk):
        self.root = root
        self.root.title("Visual Detection — Face Recognition Demo")
        self.root.geometry("1000x700")
        self.root.config(bg="#1e1e1e")
        self.root.minsize(800, 600)

        # State
        self.cv_bgr = None
        self.cv_annotated = None
        self.tk_img = None
        self.current_path: Path | None = None

        # MediaPipe face detector
        self.face_detector = FaceDetectorMP(
            min_conf=0.6,
            model_selection=1,
            pad=0.08,
        )

        # --- HEADER BAR ---
        header = ttk.Frame(self.root, padding=12)
        header.pack(side="top", fill="x")

        title_label = ttk.Label(
            header,
            text="Visual Detection — MediaPipe Face Detection",
            font=("Segoe UI", 16, "bold")
        )
        title_label.pack(side="left")

        # --- TOOLBAR ---
        toolbar = ttk.Frame(self.root, padding=10)
        toolbar.pack(side="top", fill="x")

        self.open_btn = ttk.Button(toolbar, text="Open Image", command=self.open_image)
        self.open_btn.pack(side="left", padx=5)

        self.detect_btn = ttk.Button(
            toolbar, text="Detect Faces", command=self.detect_faces, state="disabled"
        )
        self.detect_btn.pack(side="left", padx=5)

        self.save_btn = ttk.Button(
            toolbar, text="Save Annotated", command=self.save_annotated, state="disabled"
        )
        self.save_btn.pack(side="left", padx=5)

        self.status = ttk.Label(toolbar, text="Load an image to begin.")
        self.status.pack(side="right")

        # --- CANVAS AREA ---
        self.canvas_frame = ttk.Frame(self.root, padding=10)
        self.canvas_frame.pack(fill="both", expand=True)

        self.canvas = Canvas(
            self.canvas_frame, bg="#0f0f0f", highlightthickness=0
        )
        self.canvas.pack(fill="both", expand=True)

        # Redraw image when window resizes
        self.root.bind("<Configure>", lambda e: self._redraw())

        # Modern theme
        try:
            ttk.Style().theme_use("clam")
        except:
            pass

    # ---------- File Operations ----------
    def open_image(self):
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                       ("All files", "*.*")]
        )
        if not path:
            return

        self.current_path = Path(path)
        img = cv2.imread(path)

        if img is None:
            messagebox.showerror("Error", "Could not load image.")
            return

        self.cv_bgr = img
        self.cv_annotated = None

        self.detect_btn.config(state="normal")
        self.save_btn.config(state="disabled")

        self.status.config(
            text=f"Loaded: {self.current_path.name} — {img.shape[1]}x{img.shape[0]}"
        )
        self._redraw()

    def save_annotated(self):
        if self.cv_annotated is None:
            messagebox.showinfo("Info", "Run detection first.")
            return

        out_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile=(
                self.current_path.stem + "_annotated.png"
                if self.current_path else "annotated.png"
            ),
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("All files", "*.*")
            ]
        )

        if not out_path:
            return

        cv2.imwrite(out_path, self.cv_annotated)
        self.status.config(text=f"Saved: {Path(out_path).name}")

    # ---------- Detection ----------
    def detect_faces(self):
        if self.cv_bgr is None:
            return

        boxes = self.face_detector.detect(self.cv_bgr)

        annotated = self.cv_bgr.copy()
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        self.cv_annotated = annotated
        self.save_btn.config(state="normal")
        self.status.config(text=f"Faces detected: {len(boxes)}")

        self._redraw(annotated=True)

    # ---------- Drawing ----------
    def _redraw(self, annotated=False):
        self.canvas.delete("all")

        # Which image to show?
        if annotated and self.cv_annotated is not None:
            img = self.cv_annotated
        else:
            img = self.cv_bgr

        if img is None:
            self.canvas.create_text(
                self.canvas.winfo_width() // 2,
                self.canvas.winfo_height() // 2,
                text="Open an image to display",
                fill="#bbbbbb",
                font=("Segoe UI", 14),
            )
            return

        # Scale image to fit inside canvas
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        ih, iw = img.shape[:2]
        scale = min(cw / iw, ch / ih)
        new_w, new_h = int(iw * scale), int(ih * scale)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb).resize((new_w, new_h), Image.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(pil_img)

        x = (cw - new_w) // 2
        y = (ch - new_h) // 2
        self.canvas.create_image(x, y, anchor="nw", image=self.tk_img)


def main():
    root = Tk()
    VisualDetectionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

