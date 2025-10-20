import cv2
import numpy as np
from pathlib import Path
from tkinter import Tk, Canvas, filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk

# ---- Face detector (OpenCV Haar cascade) ----
FACE_CASCADE = cv2.CascadeClassifier(
    str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
)

# ---- Tkinter App ----
class VisualDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Basic Visual Detection (Tkinter + OpenCV)")
        self.root.geometry("900x650")
        self.root.minsize(700, 500)

        # State
        self.cv_bgr = None          # original image (BGR OpenCV)
        self.cv_annotated = None    # annotated image (BGR)
        self.tk_img = None          # keep ref to avoid GC
        self.current_path = None

        # UI: controls at top
        top = ttk.Frame(root, padding=8)
        top.pack(side="top", fill="x")

        ttk.Button(top, text="Open Image", command=self.open_image).pack(side="left")
        self.detect_btn = ttk.Button(top, text="Detect Faces", command=self.detect_faces, state="disabled")
        self.detect_btn.pack(side="left", padx=6)
        self.save_btn = ttk.Button(top, text="Save Annotated", command=self.save_annotated, state="disabled")
        self.save_btn.pack(side="left", padx=6)

        self.status = ttk.Label(top, text="Load an image to begin.")
        self.status.pack(side="right")

        # UI: canvas for image
        self.canvas = Canvas(root, bg="#111", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=8, pady=8)

        # Resize handler to redraw when window size changes
        self.root.bind("<Configure>", lambda e: self._redraw())

    # ---- File ops ----
    def open_image(self):
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")]
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
        self.status.config(text=f"Loaded: {self.current_path.name} — {img.shape[1]}x{img.shape[0]}")
        self._redraw()

    def save_annotated(self):
        if self.cv_annotated is None:
            messagebox.showinfo("Info", "Run detection first.")
            return
        out_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile=(self.current_path.stem + "_annotated.png") if self.current_path else "annotated.png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg *.jpeg"), ("All files", "*.*")]
        )
        if not out_path:
            return
        cv2.imwrite(out_path, self.cv_annotated)
        self.status.config(text=f"Saved: {Path(out_path).name}")

    # ---- Detection ----
    def detect_faces(self):
        if self.cv_bgr is None:
            return
        gray = cv2.cvtColor(self.cv_bgr, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

        annotated = self.cv_bgr.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

        self.cv_annotated = annotated
        self.save_btn.config(state="normal")
        self.status.config(text=f"Faces detected: {len(faces)}")
        self._redraw(annotated=True)

    # ---- Drawing ----
    def _redraw(self, annotated=False):
        self.canvas.delete("all")
        img = None
        if annotated and self.cv_annotated is not None:
            img = self.cv_annotated
        elif self.cv_bgr is not None:
            img = self.cv_bgr

        if img is None:
            w = self.canvas.winfo_width()
            h = self.canvas.winfo_height()
            self.canvas.create_text(w//2, h//2, text="Open an image to view it here", fill="#bbb", font=("Arial", 14))
            return

        cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        ih, iw = img.shape[:2]
        scale = min(cw / iw, ch / ih)
        new_w, new_h = max(1, int(iw * scale)), max(1, int(ih * scale))

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        from PIL import Image, ImageTk  # ensure imported even if file moved
        pil_img = Image.fromarray(rgb).resize((new_w, new_h), Image.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(pil_img)

        x = (cw - new_w) // 2
        y = (ch - new_h) // 2
        self.canvas.create_image(x, y, anchor="nw", image=self.tk_img)

def main():
    root = Tk()
    try:
        ttk.Style().theme_use("clam")
    except Exception:
        pass
    app = VisualDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
