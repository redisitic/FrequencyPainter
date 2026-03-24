import tkinter as tk
from tkinter import filedialog, ttk, messagebox, colorchooser
import numpy as np
from PIL import Image, ImageTk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io


# functions

def compute_fft_channel(channel_arr):
    f = np.fft.fft2(channel_arr)
    f_shift = np.fft.fftshift(f)
    log_magnitude = np.log1p(np.abs(f_shift))
    return log_magnitude, f_shift


def compute_fft_rgb(rgb_arr):
    r_log, r_shift = compute_fft_channel(rgb_arr[:, :, 0])
    g_log, g_shift = compute_fft_channel(rgb_arr[:, :, 1])
    b_log, b_shift = compute_fft_channel(rgb_arr[:, :, 2])
    global_max = max(r_log.max(), g_log.max(), b_log.max())
    def norm(x):
        return (x / global_max * 255).astype(np.uint8)
    fft_rgb = np.stack([norm(r_log), norm(g_log), norm(b_log)], axis=2)
    return Image.fromarray(fft_rgb, "RGB"), r_shift, g_shift, b_shift, global_max


def reconstruct_from_fft_display(fft_pil, r_shift, g_shift, b_shift, global_max):
    arr = np.array(fft_pil, dtype=np.float32)
    def recon(disp_ch, orig_shift):
        log_mag   = disp_ch / 255.0 * global_max
        magnitude = np.expm1(log_mag)
        phase     = np.angle(orig_shift)
        new_shift = magnitude * np.exp(1j * phase)
        spatial   = np.fft.ifft2(np.fft.ifftshift(new_shift))
        return np.clip(np.real(spatial), 0, 255).astype(np.uint8)
    r = recon(arr[:, :, 0], r_shift)
    g = recon(arr[:, :, 1], g_shift)
    b = recon(arr[:, :, 2], b_shift)
    return Image.fromarray(np.stack([r, g, b], axis=2))


def render_histogram(gray_arr, title, color):
    fig, ax = plt.subplots(figsize=(3.8, 1.8), dpi=80)
    fig.patch.set_facecolor("#000000")
    ax.set_facecolor("#000000")
    ax.hist(gray_arr.ravel(), bins=128, color=color, alpha=0.85, edgecolor="none")
    ax.set_title(title, color="white", fontsize=8, pad=3)
    ax.tick_params(colors="#555555", labelsize=6)
    for spine in ax.spines.values():
        spine.set_edgecolor("#222")
    fig.tight_layout(pad=0.4)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


# zoomable + drawable canvas

class ZoomableCanvas(tk.Canvas):
    MIN_ZOOM  = 0.1
    MAX_ZOOM  = 20.0
    ZOOM_STEP = 1.15
    MIN_BRUSH = 1
    MAX_BRUSH = 300

    def __init__(self, parent, brush_state, **kwargs):
        super().__init__(parent, bg="#000000", highlightthickness=0, **kwargs)
        self._pil_image  = None
        self._tk_image   = None
        self._zoom       = 1.0
        self._offset_x   = 0.0
        self._offset_y   = 0.0
        self._drag_start = None
        self._draw_arr   = None
        self._last_draw  = None
        self._mouse_pos  = None
        self._brush      = brush_state
        self.on_stroke_end = None  # callback(pil_image) fired on mouse release

        self.bind("<MouseWheel>",          self._on_scroll)
        self.bind("<Button-4>",            self._on_scroll)
        self.bind("<Button-5>",            self._on_scroll)
        self.bind("<Alt-ButtonPress-1>",   self._on_pan_start)
        self.bind("<Alt-B1-Motion>",       self._on_pan_move)
        self.bind("<Alt-ButtonRelease-1>", self._on_pan_end)
        self.bind("<ButtonPress-1>",       self._on_draw_start)
        self.bind("<B1-Motion>",           self._on_draw_move)
        self.bind("<ButtonRelease-1>",     self._on_draw_end)
        self.bind("<Motion>",              self._on_mouse_move)
        self.bind("<Leave>",               self._on_mouse_leave)
        self.bind("<Configure>",           lambda _: self._redraw())

    # image management

    def set_image(self, pil_img):
        self._pil_image = pil_img
        self._fit_image()

    def update_image(self, pil_img):
        self._pil_image = pil_img
        self._redraw()

    def reset_view(self):
        self._fit_image()

    def _fit_image(self):
        self.update_idletasks()
        cw, ch = self.winfo_width(), self.winfo_height()
        if cw < 2 or ch < 2 or self._pil_image is None:
            return
        iw, ih = self._pil_image.size
        self._zoom     = min(cw / iw, ch / ih)
        self._offset_x = (cw - iw * self._zoom) / 2
        self._offset_y = (ch - ih * self._zoom) / 2
        self._redraw()

    def _redraw(self):
        if self._pil_image is None:
            return
        cw, ch = self.winfo_width(), self.winfo_height()
        if cw < 2 or ch < 2:
            return
        iw, ih   = self._pil_image.size
        new_w    = max(1, int(iw * self._zoom))
        new_h    = max(1, int(ih * self._zoom))
        resample = Image.NEAREST if self._zoom > 3 else Image.LANCZOS
        resized  = self._pil_image.resize((new_w, new_h), resample)
        self._tk_image = ImageTk.PhotoImage(resized)
        self.delete("all")
        self.create_image(int(self._offset_x), int(self._offset_y),
                          anchor="nw", image=self._tk_image)
        self._draw_brush_cursor()

    def _draw_brush_cursor(self):
        b = self._brush
        if self._mouse_pos is None:
            return
        mx, my = self._mouse_pos
        # watermark cursor: show scaled bounding box
        if b["wm_active"] and b["wm_image"] is not None:
            wm = b["wm_image"]
            iw, ih = wm.size
            diameter = b["size"] * 2
            if iw >= ih:
                pw = max(1, int(diameter * self._zoom))
                ph = max(1, int(diameter * self._zoom * ih / iw))
            else:
                ph = max(1, int(diameter * self._zoom))
                pw = max(1, int(diameter * self._zoom * iw / ih))
            half_w, half_h = pw // 2, ph // 2
            self.create_rectangle(mx - half_w, my - half_h,
                                  mx + half_w, my + half_h,
                                  outline="#22c55e", width=1, dash=(4, 3))
            return
        if not b["active"]:
            return
        r = max(1, b["size"] * self._zoom)
        self.create_oval(mx - r, my - r, mx + r, my + r,
                         outline="#ffffff", width=1, dash=(3, 3))

    def _canvas_to_image(self, cx, cy):
        return (cx - self._offset_x) / self._zoom, (cy - self._offset_y) / self._zoom

    # zoom

    def _on_scroll(self, event):
        up = event.num == 4 or event.delta > 0
        direction = 1 if up else -1

        b = self._brush
        if b["f_held"] and (b["active"] or (b["wm_active"] and b["wm_image"] is not None)):
            delta    = max(1, b["size"] // 8)
            new_size = b["size"] + direction * delta
            b["size"] = max(self.MIN_BRUSH, min(self.MAX_BRUSH, new_size))
            b["wm_arr"] = None   # invalidate tile cache so next stamp rescales
            if b.get("size_changed_cb"):
                b["size_changed_cb"]()
            self._redraw()
            return

        if self._pil_image is None:
            return
        factor   = self.ZOOM_STEP if up else 1.0 / self.ZOOM_STEP
        new_zoom = max(self.MIN_ZOOM, min(self.MAX_ZOOM, self._zoom * factor))
        if new_zoom == self._zoom:
            return
        mx, my = event.x, event.y
        self._offset_x = mx - (mx - self._offset_x) * (new_zoom / self._zoom)
        self._offset_y = my - (my - self._offset_y) * (new_zoom / self._zoom)
        self._zoom     = new_zoom
        self._redraw()

    # pan

    def _on_pan_start(self, event):
        self._drag_start = (event.x, event.y)
        self.config(cursor="fleur")

    def _on_pan_move(self, event):
        if self._drag_start is None:
            return
        self._offset_x  += event.x - self._drag_start[0]
        self._offset_y  += event.y - self._drag_start[1]
        self._drag_start = (event.x, event.y)
        self._redraw()

    def _on_pan_end(self, event):
        self._drag_start = None
        self.config(cursor="")

    # draw

    def _on_draw_start(self, event):
        b = self._brush
        if b["alt_held"] or self._pil_image is None:
            return
        if not b["active"] and not (b["wm_active"] and b["wm_image"] is not None):
            return
        self._draw_arr = np.array(self._pil_image, dtype=np.float32)
        ix, iy         = self._canvas_to_image(event.x, event.y)
        self._last_draw = (ix, iy)
        self._paint_point(ix, iy)
        self._pil_image = Image.fromarray(np.clip(self._draw_arr, 0, 255).astype(np.uint8))
        self._redraw()

    def _on_draw_move(self, event):
        if self._draw_arr is None:
            return
        ix, iy  = self._canvas_to_image(event.x, event.y)
        lx, ly  = self._last_draw
        dist    = np.hypot(ix - lx, iy - ly)
        radius  = self._brush["size"]
        if dist > 0:
            steps = max(1, int(dist / max(1.0, radius * 0.4)))
            for i in range(1, steps + 1):
                t = i / steps
                self._paint_point(lx + t * (ix - lx), ly + t * (iy - ly))
        self._last_draw = (ix, iy)
        self._pil_image = Image.fromarray(np.clip(self._draw_arr, 0, 255).astype(np.uint8))
        self._redraw()

    def _on_draw_end(self, event):
        if self._draw_arr is None:
            return
        self._draw_arr  = None
        self._last_draw = None
        if self.on_stroke_end and self._pil_image:
            self.on_stroke_end(self._pil_image)

    def _paint_point(self, cx, cy):
        if self._brush["wm_active"] and self._brush["wm_image"] is not None:
            self._paint_watermark(cx, cy)
        else:
            self._paint_solid(cx, cy)

    def _paint_solid(self, cx, cy):
        h, w   = self._draw_arr.shape[:2]
        radius = self._brush["size"]
        x0 = max(0, int(cx - radius))
        x1 = min(w, int(cx + radius) + 1)
        y0 = max(0, int(cy - radius))
        y1 = min(h, int(cy + radius) + 1)
        if x0 >= x1 or y0 >= y1:
            return
        ys   = np.arange(y0, y1, dtype=np.float32).reshape(-1, 1)
        xs   = np.arange(x0, x1, dtype=np.float32).reshape(1, -1)
        mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= radius ** 2
        if not mask.any():
            return
        alpha = self._brush["intensity"]
        for c, val in enumerate(self._brush["color"]):
            ch       = self._draw_arr[y0:y1, x0:x1, c]
            ch[mask] = ch[mask] * (1 - alpha) + val * alpha

    def _get_wm_tile(self):
        """Return a float32 RGBA array of the watermark scaled to current brush size, cached."""
        b   = self._brush
        wm  = b["wm_image"]
        sz  = b["size"]
        key = (sz, wm.size)
        if b["wm_arr"] is not None and b["wm_arr_size"] == key:
            return b["wm_arr"]
        iw, ih  = wm.size
        diameter = sz * 2
        if iw >= ih:
            new_w = max(1, diameter)
            new_h = max(1, int(diameter * ih / iw))
        else:
            new_h = max(1, diameter)
            new_w = max(1, int(diameter * iw / ih))
        scaled = wm.resize((new_w, new_h), Image.LANCZOS)
        arr = np.array(scaled, dtype=np.float32)   # H x W x 4
        b["wm_arr"]      = arr
        b["wm_arr_size"] = key
        return arr

    def _paint_watermark(self, cx, cy):
        """Stamp the RGBA watermark tile centerd at (cx, cy).
        Per-pixel alpha from the RGBA image × global intensity controls blend weight.
        """
        tile    = self._get_wm_tile()          # float32 H x W x 4
        th, tw  = tile.shape[:2]
        h, w    = self._draw_arr.shape[:2]
        global_intensity = self._brush["intensity"]

        # destination rect (tile centerd on cursor)
        dst_x0 = int(cx) - tw // 2
        dst_y0 = int(cy) - th // 2
        dst_x1 = dst_x0 + tw
        dst_y1 = dst_y0 + th

        # clamp to canvas bounds and compute corresponding tile slice
        src_x0 = max(0, -dst_x0);  dst_x0 = max(0, dst_x0)
        src_y0 = max(0, -dst_y0);  dst_y0 = max(0, dst_y0)
        src_x1 = tw - max(0, dst_x1 - w);  dst_x1 = min(w, dst_x1)
        src_y1 = th - max(0, dst_y1 - h);  dst_y1 = min(h, dst_y1)

        if dst_x0 >= dst_x1 or dst_y0 >= dst_y1:
            return

        tile_crop = tile[src_y0:src_y1, src_x0:src_x1]   # float32 H' x W' x 4
        wm_rgb    = tile_crop[:, :, :3]                    # 0–255 range
        wm_alpha  = tile_crop[:, :, 3] / 255.0 * global_intensity  # 0–1

        dst_region = self._draw_arr[dst_y0:dst_y1, dst_x0:dst_x1]  # float32 H' x W' x 3

        # alpha-composite: out = wm_rgb * a + dst * (1 - a)
        a = wm_alpha[:, :, np.newaxis]
        self._draw_arr[dst_y0:dst_y1, dst_x0:dst_x1] = wm_rgb * a + dst_region * (1 - a)

    # cursor tracking

    def _on_mouse_move(self, event):
        self._mouse_pos = (event.x, event.y)
        b = self._brush
        if b["active"] or (b["wm_active"] and b["wm_image"] is not None):
            self._redraw()

    def _on_mouse_leave(self, event):
        self._mouse_pos = None
        b = self._brush
        if b["active"] or (b["wm_active"] and b["wm_image"] is not None):
            self._redraw()


# undo/redo history manager

MAX_HISTORY = 50  # max snapshots per stack

class HistoryManager:
    """
    Stores (orig_pil, fft_pil, r_shift, g_shift, b_shift, global_max) snapshots.
    Each stroke end pushes to undo stack and clears redo stack.
    """
    def __init__(self):
        self._undo_stack = []
        self._redo_stack = []

    def push(self, state):
        """Call before applying a stroke (saves the pre-stroke state)."""
        self._undo_stack.append(state)
        if len(self._undo_stack) > MAX_HISTORY:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def can_undo(self):
        return len(self._undo_stack) > 0

    def can_redo(self):
        return len(self._redo_stack) > 0

    def undo(self, current_state):
        if not self.can_undo():
            return None
        self._redo_stack.append(current_state)
        return self._undo_stack.pop()

    def redo(self, current_state):
        if not self.can_redo():
            return None
        self._undo_stack.append(current_state)
        return self._redo_stack.pop()

    def clear(self):
        self._undo_stack.clear()
        self._redo_stack.clear()


# main app

class FFTApp(tk.Tk):
    HIST_W, HIST_H = 304, 144

    def __init__(self):
        super().__init__()
        self.title("FrequencyPainter")
        self.configure(bg="#000000")
        self.resizable(True, True)

        self._orig_pil   = None
        self._fft_pil    = None
        self._r_shift    = None
        self._g_shift    = None
        self._b_shift    = None
        self._global_max = None

        self._history = HistoryManager()

        self._brush_state = {
            "active":           False,
            "color":            (255, 80, 80),
            "size":             15,
            "intensity":        1.0,
            "alt_held":         False,
            "f_held":           False,
            "size_changed_cb":  None,
            # watermark mode
            "wm_active":        False,
            "wm_image":         None,   # RGBA PIL image (source, any size)
            "wm_arr":           None,   # float32 RGBA numpy cache, rebuilt on size change
            "wm_arr_size":      None,   # (brush_size, wm_image.size) key for cache
        }

        self._build_ui()

        self._brush_state["size_changed_cb"] = self._update_size_label

        self.bind_all("<Alt_L>",            lambda e: self._brush_state.update({"alt_held": True}))
        self.bind_all("<Alt_R>",            lambda e: self._brush_state.update({"alt_held": True}))
        self.bind_all("<KeyRelease-Alt_L>", lambda e: self._brush_state.update({"alt_held": False}))
        self.bind_all("<KeyRelease-Alt_R>", lambda e: self._brush_state.update({"alt_held": False}))
        self.bind_all("<KeyPress-f>",       lambda e: self._brush_state.update({"f_held": True}))
        self.bind_all("<KeyPress-F>",       lambda e: self._brush_state.update({"f_held": True}))
        self.bind_all("<KeyRelease-f>",     lambda e: self._brush_state.update({"f_held": False}))
        self.bind_all("<KeyRelease-F>",     lambda e: self._brush_state.update({"f_held": False}))

        # Ctrl+Z = undo, Ctrl+Y / Ctrl+Shift+Z = redo
        self.bind_all("<Control-z>",        lambda e: self._do_undo())
        self.bind_all("<Control-Z>",        lambda e: self._do_undo())
        self.bind_all("<Control-y>",        lambda e: self._do_redo())
        self.bind_all("<Control-Y>",        lambda e: self._do_redo())
        self.bind_all("<Control-Shift-z>",  lambda e: self._do_redo())
        self.bind_all("<Control-Shift-Z>",  lambda e: self._do_redo())

        self.minsize(900, 700)

    # UI

    def _build_ui(self):
        toolbar = tk.Frame(self, bg="#111111", pady=8)
        toolbar.pack(fill="x", side="top")

        btn_style = dict(bg="#2563eb", fg="white", font=("Segoe UI", 10, "bold"),
                         relief="flat", cursor="hand2", padx=18, pady=6,
                         activebackground="#1d4ed8", activeforeground="white")
        sec_style = dict(bg="#1a1a1a", fg="#888888", font=("Segoe UI", 9),
                         relief="flat", cursor="hand2", padx=12, pady=6,
                         activebackground="#2a2a2a", activeforeground="white")
        dis_style = dict(bg="#0a0a0a", fg="#2a2a2a", font=("Segoe UI", 9),
                         relief="flat", padx=12, pady=6)

        tk.Button(toolbar, text="⬆  Upload Image",
                  command=self._load_image, **btn_style).pack(side="left", padx=12)
        tk.Button(toolbar, text="⟳  Reset View",
                  command=self._reset_views, **sec_style).pack(side="left", padx=(0, 6))

        tk.Frame(toolbar, bg="#2a2a2a", width=1).pack(side="left", fill="y", padx=8, pady=4)

        self._brush_btn = tk.Button(toolbar, text="🖌  Brush",
                                    command=self._toggle_brush, **sec_style)
        self._brush_btn.pack(side="left", padx=(0, 6))

        self._wm_btn = tk.Button(toolbar, text="🖼  Watermark",
                                 command=self._toggle_watermark, **sec_style)
        self._wm_btn.pack(side="left", padx=(0, 4))

        self._wm_label = tk.Label(toolbar, text="no image",
                                  bg="#111111", fg="#2a2a2a",
                                  font=("Segoe UI", 8, "italic"), cursor="hand2")
        self._wm_label.pack(side="left", padx=(0, 10))
        self._wm_label.bind("<Button-1>", lambda _: self._load_watermark())

        r, g, b = self._brush_state["color"]
        self._color_btn = tk.Button(toolbar, bg=f"#{r:02x}{g:02x}{b:02x}",
                                    width=3, relief="flat", cursor="hand2",
                                    command=self._pick_color, pady=6)
        self._color_btn.pack(side="left", padx=(0, 10))

        self._size_label = tk.Label(toolbar,
                                    text=f"⬤  {self._brush_state['size']}px",
                                    bg="#111111", fg="#888888", font=("Segoe UI", 9))
        self._size_label.pack(side="left", padx=(0, 10))

        tk.Label(toolbar, text="Intensity:", bg="#111111", fg="#888888",
                 font=("Segoe UI", 9)).pack(side="left")
        self._intensity_var = tk.IntVar(value=100)
        tk.Scale(toolbar, from_=0, to=100, orient="horizontal",
                 variable=self._intensity_var, bg="#111111", fg="#888888",
                 troughcolor="#1a1a1a", highlightthickness=0, bd=0,
                 length=110, showvalue=True, font=("Segoe UI", 7),
                 command=self._on_intensity_change).pack(side="left", padx=(4, 12))

        tk.Frame(toolbar, bg="#2a2a2a", width=1).pack(side="left", fill="y", padx=8, pady=4)

        # Undo / Redo buttons
        self._undo_btn = tk.Button(toolbar, text="↩  Undo",
                                   command=self._do_undo, **dis_style)
        self._undo_btn.pack(side="left", padx=(0, 4))

        self._redo_btn = tk.Button(toolbar, text="↪  Redo",
                                   command=self._do_redo, **dis_style)
        self._redo_btn.pack(side="left", padx=(0, 12))

        tk.Frame(toolbar, bg="#2a2a2a", width=1).pack(side="left", fill="y", padx=8, pady=4)

        tk.Label(toolbar, text="Scroll=zoom  •  Alt+drag=pan  •  F+scroll=brush size  •  Ctrl+Z/Y=undo/redo",
                 bg="#111111", fg="#444444", font=("Segoe UI", 8, "italic")).pack(side="left", padx=6)

        self._status = tk.StringVar(value="No image loaded")
        tk.Label(toolbar, textvariable=self._status, bg="#111111", fg="#4ade80",
                 font=("Segoe UI", 9, "italic")).pack(side="right", padx=16)

        content = tk.Frame(self, bg="#000000")
        content.pack(fill="both", expand=True, padx=16, pady=10)

        left = tk.Frame(content, bg="#000000")
        left.pack(side="left", fill="both", expand=True)
        self._orig_canvas, self._hist_orig = self._make_panel(
            left, "Original Image", "Pixel Intensity Distribution",
            save_cmd=self._save_orig)
        self._orig_canvas.on_stroke_end = self._on_orig_stroke_end

        right = tk.Frame(content, bg="#000000")
        right.pack(side="left", fill="both", expand=True, padx=(12, 0))
        self._fft_canvas, self._hist_fft = self._make_panel(
            right, "Frequency Domain (RGB FFT Magnitude)", "Frequency Magnitude Distribution",
            save_cmd=self._save_fft)
        self._fft_canvas.on_stroke_end = self._on_fft_stroke_end

        self._stats_frame = tk.Frame(self, bg="#111111", pady=6)
        self._stats_frame.pack(fill="x", side="bottom")
        self._stat_labels = {}
        for key in ("Size", "R DC", "G DC", "B DC", "Peak Freq (row,col)"):
            lbl = tk.Label(self._stats_frame, text=f"{key}: —",
                           bg="#111111", fg="#555555", font=("Segoe UI", 8))
            lbl.pack(side="left", padx=14)
            self._stat_labels[key] = lbl

        ttk.Style().theme_use("default")

    def _make_panel(self, parent, img_title, hist_title, save_cmd=None):
        outer = tk.Frame(parent, bg="#000000")
        outer.pack(fill="both", expand=True)

        img_frame = tk.Frame(outer, bg="#0d0d0d")
        img_frame.pack(fill="both", expand=True, pady=(0, 6))

        header = tk.Frame(img_frame, bg="#0d0d0d")
        header.pack(fill="x")
        tk.Label(header, text=img_title, bg="#0d0d0d", fg="#ffffff",
                 font=("Segoe UI", 9, "bold"), pady=4).pack(side="left", padx=8)
        if save_cmd:
            tk.Button(header, text="💾  Save", command=save_cmd,
                      bg="#1a1a1a", fg="#888888", font=("Segoe UI", 8),
                      relief="flat", cursor="hand2", padx=10, pady=3,
                      activebackground="#2a2a2a", activeforeground="white"
                      ).pack(side="right", padx=8, pady=4)

        canvas = ZoomableCanvas(img_frame, self._brush_state)
        canvas.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        placeholder = tk.Label(canvas, text="Upload an image to begin",
                               fg="#444444", bg="#000000", font=("Segoe UI", 9))
        placeholder.place(relx=0.5, rely=0.5, anchor="center")
        canvas._placeholder = placeholder

        hist_frame = tk.Frame(outer, bg="#0d0d0d")
        hist_frame.pack(fill="x", pady=(0, 8))
        tk.Label(hist_frame, text=hist_title, bg="#0d0d0d", fg="#888888",
                 font=("Segoe UI", 8), pady=2).pack()
        hist_lbl = tk.Label(hist_frame, bg="#000000", text="—", fg="#444444",
                            font=("Segoe UI", 8))
        hist_lbl.pack(fill="x", padx=6, pady=(0, 4))

        return canvas, hist_lbl

    # brush controls

    def _toggle_brush(self):
        self._brush_state["active"] = not self._brush_state["active"]
        if self._brush_state["active"]:
            self._brush_btn.config(bg="#2563eb", fg="white")
            # deactivate watermark mode
            self._brush_state["wm_active"] = False
            self._wm_btn.config(bg="#1a1a1a", fg="#888888")
        else:
            self._brush_btn.config(bg="#1a1a1a", fg="#888888")

    def _toggle_watermark(self):
        b = self._brush_state
        # if no watermark loaded yet, open picker first
        if b["wm_image"] is None:
            self._load_watermark()
            return
        b["wm_active"] = not b["wm_active"]
        if b["wm_active"]:
            self._wm_btn.config(bg="#16a34a", fg="white")
            # deactivate normal brush
            b["active"] = False
            self._brush_btn.config(bg="#1a1a1a", fg="#888888")
        else:
            self._wm_btn.config(bg="#1a1a1a", fg="#888888")

    def _load_watermark(self):
        path = filedialog.askopenfilename(
            title="Select RGBA watermark image",
            filetypes=[("PNG / TIFF with alpha", "*.png *.tiff *.tif *.webp"),
                       ("All image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp *.gif"),
                       ("All files", "*.*")])
        if not path:
            return
        try:
            img = Image.open(path).convert("RGBA")
        except Exception as e:
            messagebox.showerror("Error", f"Could not open watermark:\n{e}")
            return
        b = self._brush_state
        b["wm_image"]    = img
        b["wm_arr"]      = None   # invalidate cache
        b["wm_arr_size"] = None
        b["wm_active"]   = True
        b["active"]      = False  # turn off normal brush
        self._brush_btn.config(bg="#1a1a1a", fg="#888888")
        self._wm_btn.config(bg="#16a34a", fg="white")
        name = path.split("/")[-1].split("\\")[-1]
        short = name if len(name) <= 18 else name[:15] + "…"
        self._wm_label.config(text=short, fg="#86efac")

    def _pick_color(self):
        r, g, b    = self._brush_state["color"]
        init_color = f"#{r:02x}{g:02x}{b:02x}"
        result     = colorchooser.askcolor(color=init_color, title="Pick brush color")
        if result and result[0]:
            rgb = tuple(int(v) for v in result[0])
            self._brush_state["color"] = rgb
            self._color_btn.config(bg=f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}")

    def _on_intensity_change(self, val):
        self._brush_state["intensity"] = int(val) / 100.0

    def _update_size_label(self):
        self._size_label.config(text=f"⬤  {self._brush_state['size']}px")

    # history helpers

    def _snapshot(self):
        """Return a copy of the full current state for the history stack."""
        return {
            "orig_pil":   self._orig_pil.copy() if self._orig_pil else None,
            "fft_pil":    self._fft_pil.copy()  if self._fft_pil  else None,
            "r_shift":    self._r_shift.copy()   if self._r_shift  is not None else None,
            "g_shift":    self._g_shift.copy()   if self._g_shift  is not None else None,
            "b_shift":    self._b_shift.copy()   if self._b_shift  is not None else None,
            "global_max": self._global_max,
        }

    def _restore(self, state):
        """Apply a history snapshot back to the app."""
        self._orig_pil   = state["orig_pil"]
        self._fft_pil    = state["fft_pil"]
        self._r_shift    = state["r_shift"]
        self._g_shift    = state["g_shift"]
        self._b_shift    = state["b_shift"]
        self._global_max = state["global_max"]
        if self._orig_pil:
            self._orig_canvas.update_image(self._orig_pil)
        if self._fft_pil:
            self._fft_canvas.update_image(self._fft_pil)
        self._update_histograms()
        self._update_stats()
        self._refresh_undo_redo_buttons()

    def _refresh_undo_redo_buttons(self):
        """Enable/disable undo and redo buttons based on stack state."""
        sec_style  = dict(bg="#1a1a1a", fg="#888888", cursor="hand2")
        dis_style  = dict(bg="#0a0a0a", fg="#2a2a2a", cursor="")
        if self._history.can_undo():
            self._undo_btn.config(**sec_style)
        else:
            self._undo_btn.config(**dis_style)
        if self._history.can_redo():
            self._redo_btn.config(**sec_style)
        else:
            self._redo_btn.config(**dis_style)

    def _do_undo(self):
        if not self._history.can_undo():
            return
        prev = self._history.undo(self._snapshot())
        if prev:
            self._restore(prev)

    def _do_redo(self):
        if not self._history.can_redo():
            return
        next_state = self._history.redo(self._snapshot())
        if next_state:
            self._restore(next_state)

    # save helpers

    def _save_image(self, pil_img, default_name):
        if pil_img is None:
            messagebox.showwarning("Nothing to save", "No image is loaded yet.")
            return
        path = filedialog.asksaveasfilename(
            title="Save image",
            initialfile=default_name,
            defaultextension=".png",
            filetypes=[
                ("PNG image",  "*.png"),
                ("JPEG image", "*.jpg *.jpeg"),
                ("BMP image",  "*.bmp"),
                ("TIFF image", "*.tiff *.tif"),
                ("All files",  "*.*"),
            ],
        )
        if not path:
            return
        try:
            pil_img.save(path)
            self._status.set(f"Saved: {path.split('/')[-1]}")
        except Exception as e:
            messagebox.showerror("Save failed", f"Could not save image:\n{e}")

    def _save_orig(self):
        self._save_image(self._orig_pil, "original.png")

    def _save_fft(self):
        self._save_image(self._fft_pil, "fft_magnitude.png")

    # image loading

    def _load_image(self):
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp *.gif"),
                       ("All files", "*.*")])
        if not path:
            return
        try:
            self._orig_pil = Image.open(path).convert("RGB")
        except Exception as e:
            messagebox.showerror("Error", f"Could not open image:\n{e}")
            return
        self._status.set(f"Loaded: {path.split('/')[-1]}")
        self._history.clear()
        self._refresh_all()

    def _reset_views(self):
        self._orig_canvas.reset_view()
        self._fft_canvas.reset_view()

    # stroke callbacks — push history BEFORE applying changes

    def _on_orig_stroke_end(self, new_pil):
        """Painted on original → push history, recompute FFT, update fft canvas."""
        self._history.push(self._snapshot())
        self._orig_pil = new_pil
        self._recompute_fft()
        self._fft_canvas.update_image(self._fft_pil)
        self._update_histograms()
        self._update_stats()
        self._refresh_undo_redo_buttons()

    def _on_fft_stroke_end(self, new_fft_pil):
        """
        Painted on FFT display → push history, reconstruct original via IFFT
        (magnitude from paint, phase from stored shifts), then sync shifts.
        """
        self._history.push(self._snapshot())
        self._fft_pil = new_fft_pil
        if self._r_shift is not None:
            self._orig_pil = reconstruct_from_fft_display(
                self._fft_pil, self._r_shift, self._g_shift,
                self._b_shift, self._global_max)
            arr = np.array(self._orig_pil)
            _, self._r_shift, self._g_shift, self._b_shift, self._global_max = \
                compute_fft_rgb(arr)
        self._orig_canvas.update_image(self._orig_pil)
        self._update_histograms()
        self._update_stats()
        self._refresh_undo_redo_buttons()

    # display

    def _refresh_all(self):
        if self._orig_pil is None:
            return
        for canvas in (self._orig_canvas, self._fft_canvas):
            if hasattr(canvas, "_placeholder"):
                canvas._placeholder.place_forget()
        self._recompute_fft()
        self._orig_canvas.set_image(self._orig_pil.copy())
        self._fft_canvas.set_image(self._fft_pil)
        self._update_histograms()
        self._update_stats()
        self._refresh_undo_redo_buttons()

    def _recompute_fft(self):
        arr = np.array(self._orig_pil)
        self._fft_pil, self._r_shift, self._g_shift, self._b_shift, self._global_max = \
            compute_fft_rgb(arr)

    def _update_histograms(self):
        gray     = np.array(self._orig_pil.convert("L"))
        h_orig   = render_histogram(gray, "Original", "#2563eb")
        h_orig_tk = ImageTk.PhotoImage(h_orig.resize((self.HIST_W, self.HIST_H), Image.LANCZOS))
        self._hist_orig.configure(image=h_orig_tk, text="")
        self._hist_orig.image = h_orig_tk

        combined_mag = np.mean([
            np.log1p(np.abs(self._r_shift)),
            np.log1p(np.abs(self._g_shift)),
            np.log1p(np.abs(self._b_shift))
        ], axis=0).astype(np.uint8)
        h_fft    = render_histogram(combined_mag, "FFT Magnitude (avg RGB)", "#dc2626")
        h_fft_tk = ImageTk.PhotoImage(h_fft.resize((self.HIST_W, self.HIST_H), Image.LANCZOS))
        self._hist_fft.configure(image=h_fft_tk, text="")
        self._hist_fft.image = h_fft_tk

    def _update_stats(self):
        gray   = np.array(self._orig_pil.convert("L"))
        h, w   = gray.shape
        r_dc   = float(np.abs(self._r_shift[h // 2, w // 2]))
        g_dc   = float(np.abs(self._g_shift[h // 2, w // 2]))
        b_dc   = float(np.abs(self._b_shift[h // 2, w // 2]))
        r_mag  = np.abs(self._r_shift).copy()
        r_mag[h // 2 - 2:h // 2 + 3, w // 2 - 2:w // 2 + 3] = 0
        peak   = np.unravel_index(np.argmax(r_mag), r_mag.shape)
        self._stat_labels["Size"].config(text=f"Size: {w}×{h}")
        self._stat_labels["R DC"].config(text=f"R DC: {r_dc:,.0f}")
        self._stat_labels["G DC"].config(text=f"G DC: {g_dc:,.0f}")
        self._stat_labels["B DC"].config(text=f"B DC: {b_dc:,.0f}")
        self._stat_labels["Peak Freq (row,col)"].config(text=f"Peak Freq: ({peak[0]}, {peak[1]})")


if __name__ == "__main__":
    app = FFTApp()
    app.mainloop()