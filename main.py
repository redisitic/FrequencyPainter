import tkinter as tk
from tkinter import filedialog, ttk, messagebox, colorchooser
import numpy as np
from PIL import Image, ImageTk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from math import gcd
import io


# ── module-level helpers ───────────────────────────────────────────────────────

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


def reconstruct_from_shifts(r_shift, g_shift, b_shift):
    """IFFT the stored complex spectra directly, without going through a display image."""
    def recon(shift):
        spatial = np.fft.ifft2(np.fft.ifftshift(shift))
        return np.clip(np.real(spatial), 0, 255).astype(np.uint8)
    return Image.fromarray(np.stack([recon(r_shift), recon(g_shift), recon(b_shift)], axis=2))


def compute_phase_image_bw(r_shift, g_shift, b_shift):
    """
    Circular-mean phase of R/G/B channels → B&W image (returned as RGB PIL).

    Encoding: pixel value 0 = −pi, 128 ≈ 0, 255 = +pi.
    The circular mean (average of unit-phasors) avoids the wrap-discontinuity
    that naïve averaging produces near the ±pi boundary.
    """
    avg_complex = (
        np.exp(1j * np.angle(r_shift)) +
        np.exp(1j * np.angle(g_shift)) +
        np.exp(1j * np.angle(b_shift))
    )
    avg_angle = np.angle(avg_complex)                              # −pi … +pi
    gray = ((avg_angle + np.pi) / (2.0 * np.pi) * 255).astype(np.uint8)
    return Image.fromarray(np.stack([gray, gray, gray], axis=2), "RGB")


def angle_to_gray(angle_rad):
    """Map an angle in radians (−pi … +pi) to a uint8 gray level (0 … 255)."""
    return int(round(max(0, min(255, (angle_rad + np.pi) / (2.0 * np.pi) * 255))))


def angle_steps_label(steps):
    """
    Human-readable fraction-of-pi string for integer steps (−8 … 8),
    where 1 step = pi/8.

    Examples:  −8 → "−pi"   4 → "pi/2"   3 → "3pi/8"   0 → "0"
    """
    if steps == 0:
        return "0"
    sign = "\u2212" if steps < 0 else ""   # unicode minus
    n = abs(steps)
    if n == 8:
        return f"{sign}\u03c0"
    g = gcd(n, 8)
    num, den = n // g, 8 // g
    return f"{sign}\u03c0/{den}" if num == 1 else f"{sign}{num}\u03c0/{den}"


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


# ── zoomable + drawable canvas ────────────────────────────────────────────────

class ZoomableCanvas(tk.Canvas):
    MIN_ZOOM  = 0.1
    MAX_ZOOM  = 20.0
    ZOOM_STEP = 1.15
    MIN_BRUSH = 1
    MAX_BRUSH = 300

    def __init__(self, parent, brush_state, **kwargs):
        super().__init__(parent, bg="#000000", highlightthickness=0, **kwargs)
        self._pil_image    = None
        self._tk_image     = None
        self._zoom         = 1.0
        self._offset_x     = 0.0
        self._offset_y     = 0.0
        self._drag_start   = None
        self._draw_arr     = None
        self._last_draw    = None
        self._mouse_pos    = None
        self._brush        = brush_state
        self._redraw_job   = None
        self.on_stroke_end = None   # callback(pil_image)

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
        self._redraw_job = None
        if self._pil_image is None:
            return
        cw, ch = self.winfo_width(), self.winfo_height()
        if cw < 2 or ch < 2:
            return

        iw, ih = self._pil_image.size
        img_x0 = -self._offset_x / self._zoom
        img_y0 = -self._offset_y / self._zoom
        img_x1 = img_x0 + cw / self._zoom
        img_y1 = img_y0 + ch / self._zoom

        crop_x0 = max(0, int(img_x0))
        crop_y0 = max(0, int(img_y0))
        crop_x1 = min(iw, int(img_x1) + 1)
        crop_y1 = min(ih, int(img_y1) + 1)

        if crop_x0 >= crop_x1 or crop_y0 >= crop_y1:
            self.delete("all")
            self._draw_brush_cursor()
            return

        crop     = self._pil_image.crop((crop_x0, crop_y0, crop_x1, crop_y1))
        render_w = max(1, min(cw, int((crop_x1 - crop_x0) * self._zoom)))
        render_h = max(1, min(ch, int((crop_y1 - crop_y0) * self._zoom)))
        resample = Image.NEAREST if self._zoom > 3 else Image.LANCZOS
        resized  = crop.resize((render_w, render_h), resample)
        canvas_x = int(self._offset_x + crop_x0 * self._zoom)
        canvas_y = int(self._offset_y + crop_y0 * self._zoom)

        self._tk_image = ImageTk.PhotoImage(resized)
        self.delete("all")
        self.create_image(canvas_x, canvas_y, anchor="nw", image=self._tk_image)
        self._draw_brush_cursor()

    def _draw_brush_cursor(self):
        b = self._brush
        if self._mouse_pos is None:
            return
        mx, my = self._mouse_pos
        if b["wm_active"] and b["wm_image"] is not None:
            wm = b["wm_image"]
            iw, ih   = wm.size
            diameter = b["size"] * 2
            if iw >= ih:
                pw = max(1, int(diameter * self._zoom))
                ph = max(1, int(diameter * self._zoom * ih / iw))
            else:
                ph = max(1, int(diameter * self._zoom))
                pw = max(1, int(diameter * self._zoom * iw / ih))
            hw, hh = pw // 2, ph // 2
            self.create_rectangle(mx - hw, my - hh, mx + hw, my + hh,
                                  outline="#22c55e", width=1, dash=(4, 3))
            return
        if not b["active"]:
            return
        r        = max(1, b["size"] * self._zoom)
        hardness = b.get("hardness", 1.0)
        self.create_oval(mx - r, my - r, mx + r, my + r,
                         outline="#ffffff", width=1, dash=(3, 3))
        if hardness < 0.99:
            inner_r = max(1, r * hardness)
            self.create_oval(mx - inner_r, my - inner_r,
                             mx + inner_r, my + inner_r,
                             outline="#aaaaaa", width=1)

    def _canvas_to_image(self, cx, cy):
        return (cx - self._offset_x) / self._zoom, (cy - self._offset_y) / self._zoom

    def _on_scroll(self, event):
        up        = event.num == 4 or event.delta > 0
        direction = 1 if up else -1
        b = self._brush
        if b["f_held"] and (b["active"] or (b["wm_active"] and b["wm_image"] is not None)):
            delta    = max(1, b["size"] // 8)
            new_size = b["size"] + direction * delta
            b["size"]   = max(self.MIN_BRUSH, min(self.MAX_BRUSH, new_size))
            b["wm_arr"] = None
            if b.get("size_changed_cb"):
                b["size_changed_cb"]()
            self._schedule_redraw()
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
        self._schedule_redraw()

    def _schedule_redraw(self, delay_ms=16):
        if self._redraw_job is not None:
            self.after_cancel(self._redraw_job)
        self._redraw_job = self.after(delay_ms, self._redraw)

    def _on_pan_start(self, event):
        self._drag_start = (event.x, event.y)
        self.config(cursor="fleur")

    def _on_pan_move(self, event):
        if self._drag_start is None:
            return
        self._offset_x  += event.x - self._drag_start[0]
        self._offset_y  += event.y - self._drag_start[1]
        self._drag_start = (event.x, event.y)
        self._schedule_redraw()

    def _on_pan_end(self, event):
        self._drag_start = None
        self.config(cursor="")

    def _on_draw_start(self, event):
        b = self._brush
        if b["alt_held"] or self._pil_image is None:
            return
        if not b["active"] and not (b["wm_active"] and b["wm_image"] is not None):
            return
        self._draw_arr  = np.array(self._pil_image, dtype=np.float32)
        ix, iy          = self._canvas_to_image(event.x, event.y)
        self._last_draw = (ix, iy)
        self._paint_point(ix, iy)
        self._pil_image = Image.fromarray(np.clip(self._draw_arr, 0, 255).astype(np.uint8))
        self._redraw()

    def _on_draw_move(self, event):
        if self._draw_arr is None:
            return
        ix, iy    = self._canvas_to_image(event.x, event.y)
        lx, ly    = self._last_draw
        dist      = np.hypot(ix - lx, iy - ly)
        radius    = self._brush["size"]
        if dist > 0:
            step_size = max(1.0, radius * 0.4) if radius > 1 else 1.0
            steps = max(1, int(dist / step_size))
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
        h, w     = self._draw_arr.shape[:2]
        radius   = self._brush["size"]
        hardness = self._brush.get("hardness", 1.0)

        if radius <= 1:
            px, py = int(cx), int(cy)
            if 0 <= px < w and 0 <= py < h:
                alpha = self._brush["intensity"]
                for c, val in enumerate(self._brush["color"]):
                    old = self._draw_arr[py, px, c]
                    self._draw_arr[py, px, c] = old * (1.0 - alpha) + val * alpha
            return

        x0 = max(0, int(cx - radius));  x1 = min(w, int(cx + radius) + 1)
        y0 = max(0, int(cy - radius));  y1 = min(h, int(cy + radius) + 1)
        if x0 >= x1 or y0 >= y1:
            return

        ys   = np.arange(y0, y1, dtype=np.float32).reshape(-1, 1)
        xs   = np.arange(x0, x1, dtype=np.float32).reshape(1, -1)
        dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)

        inner_radius = hardness * radius
        feather_band = max(radius - inner_radius, 1e-6)
        falloff = np.where(
            dist <= inner_radius, 1.0,
            np.where(dist <= radius,
                     0.5 * (1.0 + np.cos(np.pi * (dist - inner_radius) / feather_band)),
                     0.0)
        ).astype(np.float32)
        alpha = falloff * self._brush["intensity"]

        for c, val in enumerate(self._brush["color"]):
            ch = self._draw_arr[y0:y1, x0:x1, c]
            ch[:] = ch * (1.0 - alpha) + val * alpha

    def _get_wm_tile(self):
        b   = self._brush
        wm  = b["wm_image"]
        sz  = b["size"]
        key = (sz, wm.size)
        if b["wm_arr"] is not None and b["wm_arr_size"] == key:
            return b["wm_arr"]
        iw, ih   = wm.size
        diameter = sz * 2
        if iw >= ih:
            new_w = max(1, diameter);  new_h = max(1, int(diameter * ih / iw))
        else:
            new_h = max(1, diameter);  new_w = max(1, int(diameter * iw / ih))
        arr = np.array(wm.resize((new_w, new_h), Image.LANCZOS), dtype=np.float32)
        b["wm_arr"] = arr;  b["wm_arr_size"] = key
        return arr

    def _paint_watermark(self, cx, cy):
        tile  = self._get_wm_tile()
        th, tw = tile.shape[:2]
        h, w   = self._draw_arr.shape[:2]
        gi     = self._brush["intensity"]

        dx0 = int(cx) - tw // 2;  dy0 = int(cy) - th // 2
        dx1 = dx0 + tw;            dy1 = dy0 + th

        sx0 = max(0, -dx0);  dx0 = max(0, dx0)
        sy0 = max(0, -dy0);  dy0 = max(0, dy0)
        sx1 = tw - max(0, dx1 - w);  dx1 = min(w, dx1)
        sy1 = th - max(0, dy1 - h);  dy1 = min(h, dy1)

        if dx0 >= dx1 or dy0 >= dy1:
            return
        tc  = tile[sy0:sy1, sx0:sx1]
        a   = (tc[:, :, 3] / 255.0 * gi)[:, :, np.newaxis]
        dst = self._draw_arr[dy0:dy1, dx0:dx1]
        self._draw_arr[dy0:dy1, dx0:dx1] = tc[:, :, :3] * a + dst * (1 - a)

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


# ── undo/redo history ─────────────────────────────────────────────────────────

MAX_HISTORY = 50

class HistoryManager:
    def __init__(self):
        self._undo_stack = []
        self._redo_stack = []

    def push(self, state):
        self._undo_stack.append(state)
        if len(self._undo_stack) > MAX_HISTORY:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def can_undo(self): return bool(self._undo_stack)
    def can_redo(self): return bool(self._redo_stack)

    def undo(self, current_state):
        if not self.can_undo(): return None
        self._redo_stack.append(current_state)
        return self._undo_stack.pop()

    def redo(self, current_state):
        if not self.can_redo(): return None
        self._undo_stack.append(current_state)
        return self._redo_stack.pop()

    def clear(self):
        self._undo_stack.clear()
        self._redo_stack.clear()


# ── main app ──────────────────────────────────────────────────────────────────

class FFTApp(tk.Tk):
    HIST_W, HIST_H = 304, 144

    def __init__(self):
        super().__init__()
        self.title("FrequencyPainter")
        self.configure(bg="#000000")
        self.resizable(True, True)

        self._orig_pil   = None
        self._fft_pil    = None
        self._phase_pil  = None
        self._r_shift    = None
        self._g_shift    = None
        self._b_shift    = None
        self._global_max = None

        self._fft_view_mode      = "magnitude"   # "magnitude" | "phase"
        self._phase_angle_steps  = 0             # integer −8 … 8; 1 step = pi/8
        self._brush_color_saved  = None          # saved while in phase mode

        self._history = HistoryManager()

        self._brush_state = {
            "active":           False,
            "color":            (255, 80, 80),
            "size":             15,
            "intensity":        1.0,
            "hardness":         1.0,
            "alt_held":         False,
            "f_held":           False,
            "size_changed_cb":  None,
            "wm_active":        False,
            "wm_image":         None,
            "wm_arr":           None,
            "wm_arr_size":      None,
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
        self.bind_all("<Control-z>",        lambda e: self._do_undo())
        self.bind_all("<Control-Z>",        lambda e: self._do_undo())
        self.bind_all("<Control-y>",        lambda e: self._do_redo())
        self.bind_all("<Control-Y>",        lambda e: self._do_redo())
        self.bind_all("<Control-Shift-z>",  lambda e: self._do_redo())
        self.bind_all("<Control-Shift-Z>",  lambda e: self._do_redo())

        self.minsize(900, 700)

    # ── UI ────────────────────────────────────────────────────────────────────

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

        tk.Button(toolbar, text="\u2b06  Upload Image",
                  command=self._load_image, **btn_style).pack(side="left", padx=12)
        tk.Button(toolbar, text="\u27f3  Reset View",
                  command=self._reset_views, **sec_style).pack(side="left", padx=(0, 6))
        tk.Frame(toolbar, bg="#2a2a2a", width=1).pack(side="left", fill="y", padx=8, pady=4)

        self._brush_btn = tk.Button(toolbar, text="\U0001f58c  Brush",
                                    command=self._toggle_brush, **sec_style)
        self._brush_btn.pack(side="left", padx=(0, 6))

        self._wm_btn = tk.Button(toolbar, text="\U0001f5bc  Watermark",
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

        self._size_frame = tk.Frame(toolbar, bg="#111111")
        self._size_frame.pack(side="left", padx=(0, 10))
        self._size_label = tk.Label(self._size_frame,
                                    text=f"\u2b24  {self._brush_state['size']}px",
                                    bg="#111111", fg="#888888", font=("Segoe UI", 9),
                                    cursor="xterm")
        self._size_label.pack()
        self._size_label.bind("<Button-1>", lambda _: self._begin_size_edit())
        self._size_var   = tk.StringVar(value=str(self._brush_state["size"]))
        self._size_entry = tk.Entry(self._size_frame, textvariable=self._size_var,
                                    width=4, bg="#1a1a1a", fg="white",
                                    insertbackground="white", relief="flat",
                                    font=("Segoe UI", 9), justify="center")
        self._size_entry.bind("<Return>",   lambda _: self._commit_size_entry())
        self._size_entry.bind("<FocusOut>", lambda _: self._commit_size_entry())
        self._size_entry.bind("<Escape>",   lambda _: self._abort_size_entry())

        tk.Label(toolbar, text="Intensity:", bg="#111111", fg="#888888",
                 font=("Segoe UI", 9)).pack(side="left")
        self._intensity_var = tk.IntVar(value=100)
        tk.Scale(toolbar, from_=0, to=100, orient="horizontal",
                 variable=self._intensity_var, bg="#111111", fg="#888888",
                 troughcolor="#1a1a1a", highlightthickness=0, bd=0,
                 length=110, showvalue=True, font=("Segoe UI", 7),
                 command=self._on_intensity_change).pack(side="left", padx=(4, 12))

        tk.Label(toolbar, text="Hardness:", bg="#111111", fg="#888888",
                 font=("Segoe UI", 9)).pack(side="left")
        self._hardness_var = tk.IntVar(value=100)
        tk.Scale(toolbar, from_=0, to=100, orient="horizontal",
                 variable=self._hardness_var, bg="#111111", fg="#888888",
                 troughcolor="#1a1a1a", highlightthickness=0, bd=0,
                 length=110, showvalue=True, font=("Segoe UI", 7),
                 command=self._on_hardness_change).pack(side="left", padx=(4, 12))

        tk.Frame(toolbar, bg="#2a2a2a", width=1).pack(side="left", fill="y", padx=8, pady=4)
        self._undo_btn = tk.Button(toolbar, text="\u21a9  Undo",
                                   command=self._do_undo, **dis_style)
        self._undo_btn.pack(side="left", padx=(0, 4))
        self._redo_btn = tk.Button(toolbar, text="\u21aa  Redo",
                                   command=self._do_redo, **dis_style)
        self._redo_btn.pack(side="left", padx=(0, 12))

        tk.Frame(toolbar, bg="#2a2a2a", width=1).pack(side="left", fill="y", padx=8, pady=4)
        tk.Label(toolbar,
                 text="Scroll=zoom  \u2022  Alt+drag=pan  \u2022  F+scroll=brush size"
                      "  \u2022  Ctrl+Z/Y=undo/redo",
                 bg="#111111", fg="#444444",
                 font=("Segoe UI", 8, "italic")).pack(side="left", padx=6)

        self._status = tk.StringVar(value="No image loaded")
        tk.Label(toolbar, textvariable=self._status, bg="#111111", fg="#4ade80",
                 font=("Segoe UI", 9, "italic")).pack(side="right", padx=16)

        # ── content area ──────────────────────────────────────────────────────
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

        # FFT panel — request both header and img_frame so we can inject controls.
        (self._fft_canvas, self._hist_fft,
         self._fft_header, self._fft_img_frame) = self._make_panel(
            right, "Frequency Domain (RGB FFT Magnitude)",
            "Frequency Magnitude Distribution",
            save_cmd=self._save_fft,
            return_header=True, return_img_frame=True)
        self._fft_canvas.on_stroke_end = self._on_fft_stroke_end

        # Phase-toggle button lives in the FFT panel header row.
        self._phase_btn = tk.Button(
            self._fft_header, text="\u25d1  Phase View",
            command=self._toggle_fft_view,
            bg="#1a1a1a", fg="#888888", font=("Segoe UI", 8),
            relief="flat", cursor="hand2", padx=10, pady=3,
            activebackground="#2a2a2a", activeforeground="white")
        self._phase_btn.pack(side="right", padx=(4, 4), pady=4)

        # ── Phase controls bar ────────────────────────────────────────────────
        # Hidden by default; revealed between the header and canvas via
        # pack(after=_fft_header) when entering phase mode.
        self._phase_controls_frame = tk.Frame(
            self._fft_img_frame, bg="#0d0d0d", pady=5, padx=10)
        # (not packed yet)

        tk.Label(self._phase_controls_frame, text="Paint angle:",
                 bg="#0d0d0d", fg="#888888",
                 font=("Segoe UI", 9)).pack(side="left", padx=(0, 6))

        self._angle_var = tk.IntVar(value=0)
        tk.Scale(self._phase_controls_frame,
                 from_=-8, to=8, resolution=1,
                 orient="horizontal",
                 variable=self._angle_var,
                 bg="#0d0d0d", fg="#888888",
                 troughcolor="#222222", highlightthickness=0, bd=0,
                 length=220, showvalue=False,
                 command=self._on_angle_change).pack(side="left", padx=(0, 8))

        # Readable label showing selected angle as fraction of pi.
        self._angle_value_label = tk.Label(
            self._phase_controls_frame, text="0",
            bg="#0d0d0d", fg="#ffffff",
            font=("Segoe UI", 9, "bold"), width=7, anchor="w")
        self._angle_value_label.pack(side="left", padx=(0, 10))

        # Gray swatch — shows exactly which shade the brush will paint.
        tk.Label(self._phase_controls_frame, text="swatch:",
                 bg="#0d0d0d", fg="#555555",
                 font=("Segoe UI", 8, "italic")).pack(side="left")
        self._angle_swatch = tk.Label(
            self._phase_controls_frame, text="  ",
            bg="#808080", width=3, relief="flat", pady=4)
        self._angle_swatch.pack(side="left", padx=(4, 14))

        tk.Label(self._phase_controls_frame,
                 text="brush writes angle to spectrum  \u2022  IFFT updates original",
                 bg="#0d0d0d", fg="#444444",
                 font=("Segoe UI", 8, "italic")).pack(side="left")

        # ── stats bar ─────────────────────────────────────────────────────────
        self._stats_frame = tk.Frame(self, bg="#111111", pady=6)
        self._stats_frame.pack(fill="x", side="bottom")
        self._stat_labels = {}
        for key in ("Size", "R DC", "G DC", "B DC", "Peak Freq (row,col)"):
            lbl = tk.Label(self._stats_frame, text=f"{key}: \u2014",
                           bg="#111111", fg="#555555", font=("Segoe UI", 8))
            lbl.pack(side="left", padx=14)
            self._stat_labels[key] = lbl

        ttk.Style().theme_use("default")

    def _make_panel(self, parent, img_title, hist_title,
                    save_cmd=None, return_header=False, return_img_frame=False):
        outer = tk.Frame(parent, bg="#000000")
        outer.pack(fill="both", expand=True)

        img_frame = tk.Frame(outer, bg="#0d0d0d")
        img_frame.pack(fill="both", expand=True, pady=(0, 6))

        header = tk.Frame(img_frame, bg="#0d0d0d")
        header.pack(fill="x")
        tk.Label(header, text=img_title, bg="#0d0d0d", fg="#ffffff",
                 font=("Segoe UI", 9, "bold"), pady=4).pack(side="left", padx=8)
        if save_cmd:
            tk.Button(header, text="\U0001f4be  Save", command=save_cmd,
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
        hist_lbl = tk.Label(hist_frame, bg="#000000", text="\u2014", fg="#444444",
                            font=("Segoe UI", 8))
        hist_lbl.pack(fill="x", padx=6, pady=(0, 4))

        if return_header and return_img_frame:
            return canvas, hist_lbl, header, img_frame
        if return_header:
            return canvas, hist_lbl, header
        if return_img_frame:
            return canvas, hist_lbl, img_frame
        return canvas, hist_lbl

    # ── phase-view toggle ─────────────────────────────────────────────────────

    def _toggle_fft_view(self):
        if self._r_shift is None:
            return

        if self._fft_view_mode == "magnitude":
            # ── enter phase mode ──────────────────────────────────────────────
            self._fft_view_mode = "phase"

            self._phase_pil = compute_phase_image_bw(
                self._r_shift, self._g_shift, self._b_shift)
            self._fft_canvas.update_image(self._phase_pil)

            # Lock the brush color to the selected angle gray.
            self._brush_color_saved = self._brush_state["color"]
            self._sync_brush_to_angle()

            # Slide the controls bar in between the header and the canvas.
            self._phase_controls_frame.pack(after=self._fft_header, fill="x")

            self._phase_btn.config(text="\u2263  Magnitude View",
                                   bg="#7c3aed", fg="white",
                                   activebackground="#6d28d9")
            self._set_fft_panel_title(
                "Frequency Domain \u2014 Phase  (B&W: 0 = \u2212\u03c0,  255 = +\u03c0)")
            self._update_phase_histogram()
            self._status.set(
                "Phase view  \u2022  brush writes selected angle into spectrum"
                "  \u2022  IFFT updates original")

        else:
            # ── return to magnitude mode ──────────────────────────────────────
            self._fft_view_mode = "magnitude"

            if self._fft_pil:
                self._fft_canvas.update_image(self._fft_pil)

            # Restore the saved brush color and re-enable the color picker.
            if self._brush_color_saved is not None:
                self._brush_state["color"] = self._brush_color_saved
                r, g, b = self._brush_color_saved
                self._color_btn.config(bg=f"#{r:02x}{g:02x}{b:02x}",
                                       state="normal", cursor="hand2")
                self._brush_color_saved = None

            self._phase_controls_frame.pack_forget()
            self._phase_btn.config(text="\u25d1  Phase View",
                                   bg="#1a1a1a", fg="#888888",
                                   activebackground="#2a2a2a")
            self._set_fft_panel_title("Frequency Domain (RGB FFT Magnitude)")
            self._update_magnitude_histogram()
            self._status.set(
                "Magnitude view  \u2022  brush edits apply IFFT reconstruction")

    def _set_fft_panel_title(self, text):
        """Update the title label in the FFT panel's header row."""
        for widget in self._fft_header.winfo_children():
            if isinstance(widget, tk.Label):
                widget.config(text=text)
                return

    # ── angle slider ──────────────────────────────────────────────────────────

    def _on_angle_change(self, val):
        steps = int(float(val))
        self._phase_angle_steps = steps
        self._sync_brush_to_angle()
        self._angle_value_label.config(text=angle_steps_label(steps))

    def _sync_brush_to_angle(self):
        """Set brush color and UI swatch to the gray encoding of the current angle."""
        angle = self._phase_angle_steps * np.pi / 8.0
        gray  = angle_to_gray(angle)
        self._brush_state["color"] = (gray, gray, gray)
        hex_c = f"#{gray:02x}{gray:02x}{gray:02x}"
        # Color button reflects the current paint shade; disable picker in phase mode.
        self._color_btn.config(bg=hex_c, state="disabled", cursor="")
        if hasattr(self, "_angle_swatch"):
            self._angle_swatch.config(bg=hex_c)

    # ── FFT display helper ────────────────────────────────────────────────────

    def _show_fft_display(self, use_set=False):
        if self._fft_view_mode == "phase" and self._r_shift is not None:
            self._phase_pil = compute_phase_image_bw(
                self._r_shift, self._g_shift, self._b_shift)
            img = self._phase_pil
        else:
            img = self._fft_pil
        if img is None:
            return
        if use_set:
            self._fft_canvas.set_image(img)
        else:
            self._fft_canvas.update_image(img)

    # ── brush controls ────────────────────────────────────────────────────────

    def _toggle_brush(self):
        self._brush_state["active"] = not self._brush_state["active"]
        if self._brush_state["active"]:
            self._brush_btn.config(bg="#2563eb", fg="white")
            self._brush_state["wm_active"] = False
            self._wm_btn.config(bg="#1a1a1a", fg="#888888")
        else:
            self._brush_btn.config(bg="#1a1a1a", fg="#888888")

    def _toggle_watermark(self):
        b = self._brush_state
        if b["wm_image"] is None:
            self._load_watermark()
            return
        b["wm_active"] = not b["wm_active"]
        if b["wm_active"]:
            self._wm_btn.config(bg="#16a34a", fg="white")
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
        b["wm_image"] = img;  b["wm_arr"] = None;  b["wm_arr_size"] = None
        b["wm_active"] = True;  b["active"] = False
        self._brush_btn.config(bg="#1a1a1a", fg="#888888")
        self._wm_btn.config(bg="#16a34a", fg="white")
        name  = path.split("/")[-1].split("\\")[-1]
        short = name if len(name) <= 18 else name[:15] + "\u2026"
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

    def _on_hardness_change(self, val):
        self._brush_state["hardness"] = int(val) / 100.0

    def _update_size_label(self):
        sz = self._brush_state["size"]
        self._size_label.config(text=f"\u2b24  {sz}px")
        self._size_var.set(str(sz))

    def _begin_size_edit(self):
        self._size_var.set(str(self._brush_state["size"]))
        self._size_label.pack_forget()
        self._size_entry.pack()
        self._size_entry.focus_set()
        self._size_entry.select_range(0, "end")

    def _commit_size_entry(self):
        try:
            val = int(self._size_var.get())
            val = max(ZoomableCanvas.MIN_BRUSH, min(ZoomableCanvas.MAX_BRUSH, val))
            self._brush_state["size"]   = val
            self._brush_state["wm_arr"] = None
        except ValueError:
            pass
        self._size_entry.pack_forget()
        self._update_size_label()
        self._size_label.pack()

    def _abort_size_entry(self):
        self._size_entry.pack_forget()
        self._update_size_label()
        self._size_label.pack()

    # ── history helpers ───────────────────────────────────────────────────────

    def _snapshot(self):
        return {
            "orig_pil":   self._orig_pil.copy() if self._orig_pil else None,
            "fft_pil":    self._fft_pil.copy()  if self._fft_pil  else None,
            "r_shift":    self._r_shift.copy()   if self._r_shift  is not None else None,
            "g_shift":    self._g_shift.copy()   if self._g_shift  is not None else None,
            "b_shift":    self._b_shift.copy()   if self._b_shift  is not None else None,
            "global_max": self._global_max,
        }

    def _restore(self, state):
        self._orig_pil   = state["orig_pil"]
        self._fft_pil    = state["fft_pil"]
        self._r_shift    = state["r_shift"]
        self._g_shift    = state["g_shift"]
        self._b_shift    = state["b_shift"]
        self._global_max = state["global_max"]
        if self._orig_pil:
            self._orig_canvas.update_image(self._orig_pil)
        self._show_fft_display(use_set=False)
        self._update_histograms()
        self._update_stats()
        self._refresh_undo_redo_buttons()

    def _refresh_undo_redo_buttons(self):
        sec = dict(bg="#1a1a1a", fg="#888888", cursor="hand2")
        dis = dict(bg="#0a0a0a", fg="#2a2a2a", cursor="")
        self._undo_btn.config(**(sec if self._history.can_undo() else dis))
        self._redo_btn.config(**(sec if self._history.can_redo() else dis))

    def _do_undo(self):
        if not self._history.can_undo():
            return
        prev = self._history.undo(self._snapshot())
        if prev:
            self._restore(prev)

    def _do_redo(self):
        if not self._history.can_redo():
            return
        nxt = self._history.redo(self._snapshot())
        if nxt:
            self._restore(nxt)

    # ── save helpers ──────────────────────────────────────────────────────────

    def _save_image(self, pil_img, default_name):
        if pil_img is None:
            messagebox.showwarning("Nothing to save", "No image is loaded yet.")
            return
        path = filedialog.asksaveasfilename(
            title="Save image", initialfile=default_name,
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("JPEG image", "*.jpg *.jpeg"),
                       ("BMP image", "*.bmp"), ("TIFF image", "*.tiff *.tif"),
                       ("All files", "*.*")])
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
        if self._fft_view_mode == "phase":
            self._save_image(self._phase_pil, "fft_phase.png")
        else:
            self._save_image(self._fft_pil, "fft_magnitude.png")

    # ── image loading ─────────────────────────────────────────────────────────

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
        if self._fft_view_mode == "phase":
            # Cleanly exit phase mode before loading.
            self._fft_view_mode = "magnitude"
            self._phase_controls_frame.pack_forget()
            self._phase_btn.config(text="\u25d1  Phase View",
                                   bg="#1a1a1a", fg="#888888",
                                   activebackground="#2a2a2a")
            self._set_fft_panel_title("Frequency Domain (RGB FFT Magnitude)")
            if self._brush_color_saved is not None:
                self._brush_state["color"] = self._brush_color_saved
                r, g, b = self._brush_color_saved
                self._color_btn.config(bg=f"#{r:02x}{g:02x}{b:02x}",
                                       state="normal", cursor="hand2")
                self._brush_color_saved = None
        self._refresh_all()

    def _reset_views(self):
        self._orig_canvas.reset_view()
        self._fft_canvas.reset_view()

    # ── stroke callbacks ──────────────────────────────────────────────────────

    def _on_orig_stroke_end(self, new_pil):
        self._history.push(self._snapshot())
        self._orig_pil = new_pil
        self._recompute_fft()
        self._show_fft_display(use_set=False)
        self._update_histograms()
        self._update_stats()
        self._refresh_undo_redo_buttons()

    def _on_fft_stroke_end(self, new_pil):
        """
        Dispatch to the correct handler depending on current view mode.

        Magnitude mode: decode painted display image → IFFT → update original.
        Phase mode:     decode painted grayscale → write angles into complex
                        shifts → IFFT → update original.
        """
        if self._fft_view_mode == "phase":
            self._phase_stroke_apply(new_pil)
        else:
            self._magnitude_stroke_apply(new_pil)

    def _magnitude_stroke_apply(self, new_fft_pil):
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

    def _phase_stroke_apply(self, painted_pil):
        """
        Write the brush stroke from the phase view back into the FFT spectrum.

        Mechanism
        ---------
        The phase image encodes phase angle as a gray level: 0 = -pi, 255 = +pi.
        When the user brushes over a region the gray values of those pixels change
        to match the selected angle (possibly with a feathered blend at the edge).

        1. Detect changed pixels by diffing the painted image against the
           pre-stroke _phase_pil.
        2. Decode each changed pixel's new gray value back to an angle in radians.
        3. For those pixels, replace the phase component in _r_shift / _g_shift /
           _b_shift while leaving the magnitude unchanged.
        4. Reconstruct _orig_pil via IFFT of the modified shifts.
        5. Build _fft_pil and _phase_pil DIRECTLY from the modified shifts —
           do NOT re-run compute_fft_rgb on the reconstructed spatial image.
           Re-FFTing the clipped uint8 spatial image scrambles the exact phases
           that were just written in: quantisation and clipping in the spatial
           domain alter the frequency-domain representation unpredictably.
           Skipping that roundtrip is what makes the painted stroke stick.
        """
        if self._phase_pil is None or self._r_shift is None:
            return

        pre_gray = np.array(self._phase_pil, dtype=np.float32)[:, :, 0]
        new_gray = np.array(painted_pil,      dtype=np.float32)[:, :, 0]

        # A 0.5-count threshold avoids false triggers from display rounding.
        mask = np.abs(new_gray - pre_gray) > 0.5
        if not mask.any():
            return   # nothing actually changed; skip history push

        self._history.push(self._snapshot())

        # Decode painted gray levels → target angles.
        # Feathered brush edges produce intermediate gray values, which decode
        # to intermediate angles, creating smooth phase transitions automatically.
        new_angles = new_gray / 255.0 * 2.0 * np.pi - np.pi

        # Apply: keep magnitude, replace phase at painted pixels.
        for shift in (self._r_shift, self._g_shift, self._b_shift):
            mag           = np.abs(shift)
            current_phase = np.angle(shift)
            updated_phase = np.where(mask, new_angles, current_phase)
            shift[:]      = mag * np.exp(1j * updated_phase)

        # Rebuild original image from modified spectrum.
        self._orig_pil = reconstruct_from_shifts(
            self._r_shift, self._g_shift, self._b_shift)

        # Build the magnitude display image directly from the modified shifts.
        # This avoids the lossy re-FFT roundtrip that would corrupt the phases.
        r_log = np.log1p(np.abs(self._r_shift))
        g_log = np.log1p(np.abs(self._g_shift))
        b_log = np.log1p(np.abs(self._b_shift))
        gmax  = max(r_log.max(), g_log.max(), b_log.max())
        self._global_max = gmax
        def _norm(x):
            return (x / gmax * 255).astype(np.uint8)
        self._fft_pil = Image.fromarray(
            np.stack([_norm(r_log), _norm(g_log), _norm(b_log)], axis=2), "RGB")

        # Use the painted image directly as the new phase display.
        # Do NOT reconstruct via compute_phase_image_bw(shifts) here: that path
        # calls np.angle(exp(i×angle)), and np.angle maps both −π and +π to +π
        # (its range is (−π, +π]). So a stroke painted at angle −π (gray=0, black)
        # would be displayed as +π (gray=255, white) — a full inversion.
        # The painted_pil is already the correct ground truth: whatever gray the
        # brush laid down is exactly what should be shown.
        self._phase_pil = painted_pil.copy()
        self._fft_canvas.update_image(self._phase_pil)
        self._orig_canvas.update_image(self._orig_pil)
        self._update_histograms()
        self._update_stats()
        self._refresh_undo_redo_buttons()

    # ── display ───────────────────────────────────────────────────────────────

    def _refresh_all(self):
        if self._orig_pil is None:
            return
        for canvas in (self._orig_canvas, self._fft_canvas):
            if hasattr(canvas, "_placeholder"):
                canvas._placeholder.place_forget()
        self._recompute_fft()
        self._orig_canvas.set_image(self._orig_pil.copy())
        self._show_fft_display(use_set=True)
        self._update_histograms()
        self._update_stats()
        self._refresh_undo_redo_buttons()

    def _recompute_fft(self):
        arr = np.array(self._orig_pil)
        self._fft_pil, self._r_shift, self._g_shift, self._b_shift, self._global_max = \
            compute_fft_rgb(arr)

    # ── histograms ────────────────────────────────────────────────────────────

    def _update_histograms(self):
        gray      = np.array(self._orig_pil.convert("L"))
        h_orig    = render_histogram(gray, "Original", "#2563eb")
        h_orig_tk = ImageTk.PhotoImage(h_orig.resize((self.HIST_W, self.HIST_H), Image.LANCZOS))
        self._hist_orig.configure(image=h_orig_tk, text="")
        self._hist_orig.image = h_orig_tk

        if self._fft_view_mode == "phase":
            self._update_phase_histogram()
        else:
            self._update_magnitude_histogram()

    def _update_magnitude_histogram(self):
        combined_mag = np.mean([
            np.log1p(np.abs(self._r_shift)),
            np.log1p(np.abs(self._g_shift)),
            np.log1p(np.abs(self._b_shift))
        ], axis=0).astype(np.uint8)
        h    = render_histogram(combined_mag, "FFT Magnitude (avg RGB)", "#dc2626")
        h_tk = ImageTk.PhotoImage(h.resize((self.HIST_W, self.HIST_H), Image.LANCZOS))
        self._hist_fft.configure(image=h_tk, text="")
        self._hist_fft.image = h_tk

    def _update_phase_histogram(self):
        """
        Distribution of circular-mean phase angle as a plain B&W bar chart,
        matching the B&W display convention used in most research papers.
        X-axis: −180 … +180 degrees, 72 bins (5 degrees each).
        """
        if self._r_shift is None:
            return
        avg_complex = (
            np.exp(1j * np.angle(self._r_shift)) +
            np.exp(1j * np.angle(self._g_shift)) +
            np.exp(1j * np.angle(self._b_shift))
        )
        angles_deg = np.degrees(np.angle(avg_complex))

        fig, ax = plt.subplots(figsize=(3.8, 1.8), dpi=80)
        fig.patch.set_facecolor("#000000")
        ax.set_facecolor("#000000")
        ax.hist(angles_deg.ravel(), bins=72, range=(-180, 180),
                color="#aaaaaa", alpha=0.85, edgecolor="none")
        ax.set_title("Phase distribution  (\u2212180\u00b0 \u2026 +180\u00b0)",
                     color="white", fontsize=8, pad=3)
        ax.set_xlabel("degrees", color="#555555", fontsize=6)
        ax.tick_params(colors="#555555", labelsize=6)
        for spine in ax.spines.values():
            spine.set_edgecolor("#222")
        fig.tight_layout(pad=0.4)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        h    = Image.open(buf).copy()
        h_tk = ImageTk.PhotoImage(h.resize((self.HIST_W, self.HIST_H), Image.LANCZOS))
        self._hist_fft.configure(image=h_tk, text="")
        self._hist_fft.image = h_tk

    def _update_stats(self):
        gray   = np.array(self._orig_pil.convert("L"))
        h, w   = gray.shape
        r_dc   = float(np.abs(self._r_shift[h // 2, w // 2]))
        g_dc   = float(np.abs(self._g_shift[h // 2, w // 2]))
        b_dc   = float(np.abs(self._b_shift[h // 2, w // 2]))
        r_mag  = np.abs(self._r_shift).copy()
        r_mag[h // 2 - 2:h // 2 + 3, w // 2 - 2:w // 2 + 3] = 0
        peak   = np.unravel_index(np.argmax(r_mag), r_mag.shape)
        self._stat_labels["Size"].config(text=f"Size: {w}\u00d7{h}")
        self._stat_labels["R DC"].config(text=f"R DC: {r_dc:,.0f}")
        self._stat_labels["G DC"].config(text=f"G DC: {g_dc:,.0f}")
        self._stat_labels["B DC"].config(text=f"B DC: {b_dc:,.0f}")
        self._stat_labels["Peak Freq (row,col)"].config(
            text=f"Peak Freq: ({peak[0]}, {peak[1]})")


if __name__ == "__main__":
    app = FFTApp()
    app.mainloop()