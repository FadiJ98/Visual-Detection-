import cv2
import numpy as np
from pathlib import Path
from tkinter import Tk, Canvas, filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk

from detection_mediapipe import FaceDetectorMP  # <- MediaPipe detector


class VisualDetectionApp:
    def __init__(self, root: Tk):
        self.root = root
        self.root.title("Basic Visual Detection (Tkinter + MediaPipe)")
        self.root.geometry("900x650")
        self.root.minsize(700, 500)

        # State
        self.cv_bgr: np.ndarray | None = None       # original image (BGR)
        self.cv_annotated: np.ndarray | None = None # annotated image (BGR)
        self.tk_img: ImageTk.PhotoImage | None = None
        self.current_path: Path | None = None

        # MediaPipe face detector (good for near + far faces)
        self.face_detector = FaceDetectorMP(
            min_conf=0.6,       # adjust if needed
            model_selection=1,  # 1 = better for further-away faces
            pad=0.08,
        )

        # --- UI: controls at top ---
        top = ttk.Frame(root, padding=8)
        top.pack(side="top", fill="x")

        ttk.Button(top, text="Open Image", command=self.open_image).pack(side="left")
        self.detect_btn = ttk.Button(
            top, text="Detect Faces", command=self.detect_faces, state="disabled"
        )
        self.detect_btn.pack(side="left", padx=6)
        self.save_btn = ttk.Button(
            top, text="Save Annotated", command=self.save_annotated, state="disabled"
        )
        self.save_btn.pack(side="left", padx=6)

        self.status = ttk.Label(top, text="Load an image to begin.")
        self.status.pack(side="right")

        # UI: canvas for image
        self.canvas = Canvas(root, bg="#111", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=8, pady=8)

        # Resize handler to redraw when window size changes
        self.root.bind("<Configure>", lambda e: self._redraw())

    # ---------- File ops ----------

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
            messagebox.showerror("Error", "Could not open image.")
            return

        self.cv_bgr = img
        self.cv_annotated = None
        self.detect_btn.config(state="normal")
        self.save_btn.config(state="disabled")
        self.status.config(
            text=f"Loaded: {self.current_path.name} — {img.shape[1]}x{img.shape[0]}"
        )
        self._redraw()

    def save_annotated(self) -> None:
        if self.cv_annotated is None:
            messagebox.showinfo("Info", "Run detection first.")
            return

        out_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile=(
                self.current_path.stem + "_annotated.png"
                if self.current_path
                else "annotated.png"
            ),
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg *.jpeg"), ("All files", "*.*")],
        )
        if not out_path:
            return

        cv2.imwrite(out_path, self.cv_annotated)
        self.status.config(text=f"Saved: {Path(out_path).name}")

    # ---------- Detection ----------

    def detect_faces(self) -> None:
        if self.cv_bgr is None:
            return

        # MediaPipe detection
        boxes = self.face_detector.detect(self.cv_bgr)

        annotated = self.cv_bgr.copy()
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        self.cv_annotated = annotated
        self.save_btn.config(state="normal")
        self.status.config(text=f"Faces detected: {len(boxes)}")
        self._redraw(annotated=True)

    # ---------- Drawing ----------

    def _redraw(self, annotated: bool = False) -> None:
        self.canvas.delete("all")

        img = None
        if annotated and self.cv_annotated is not None:
            img = self.cv_annotated
        elif self.cv_bgr is not None:
            img = self.cv_bgr

        if img is None:
            w = self.canvas.winfo_width()
            h = self.canvas.winfo_height()
            self.canvas.create_text(
                w // 2,
                h // 2,
                text="Open an image to view it here",
                fill="#bbb",
                font=("Arial", 14),
            )
            return

        cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        ih, iw = img.shape[:2]
        scale = min(cw / iw, ch / ih)
        new_w, new_h = max(1, int(iw * scale)), max(1, int(ih * scale))

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb).resize((new_w, new_h), Image.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(pil_img)

        x = (cw - new_w) // 2
        y = (ch - new_h) // 2
        self.canvas.create_image(x, y, anchor="nw", image=self.tk_img)


def main() -> None:
    root = Tk()
    try:
        ttk.Style().theme_use("clam")
    except Exception:
        pass
    app = VisualDetectionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
