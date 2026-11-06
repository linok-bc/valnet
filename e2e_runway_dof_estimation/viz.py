"""Replay frames and video for a pipeline Result.

Layout (1920 x 1080):

    +-------------------------------+------------------------+
    | camera: DeepLab mask (teal),  | cross-track vs range   |
    | label (white), runway drawn   +------------------------+
    | from the estimate (amber)     | yaw vs range           |
    +-------------------------------+------------------------+
    | readout: flight, weather, range, estimate vs truth      |
    +---------------------------------------------------------+
"""

import subprocess

import cv2
import numpy as np

W, H, CAM_W, CAM_H = 1920, 1080, 1280, 720
TEAL, WHITE, AMBER = (122, 175, 27), (245, 245, 245), (40, 180, 255)
INK, GREY, DIM, GRID, BG = (240, 240, 240), (150, 150, 150), (70, 70, 70), (45, 45, 45), (24, 24, 24)


def text(img, s, xy, scale=0.6, color=INK, thick=1):
    cv2.putText(img, s, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


class Plot:
    """A series against range (km), range decreasing to the right."""

    def __init__(self, x0, y0, w, h, title, lo, hi, rmax, ticks=None):
        self.x0, self.y0, self.w, self.h, self.title, self.lo, self.hi, self.rmax = x0, y0, w, h, title, lo, hi, rmax
        self.ticks = np.linspace(lo, hi, 5) if ticks is None else ticks

    def X(self, r):
        return int(self.x0 + (self.rmax - min(r / 1000, self.rmax)) / self.rmax * self.w)   # r in m, rmax in km

    def Y(self, v):
        return int(self.y0 + self.h - (min(max(v, self.lo), self.hi) - self.lo) / (self.hi - self.lo) * self.h)

    def draw(self, img, series):
        text(img, self.title, (self.x0, self.y0 - 12))
        cv2.rectangle(img, (self.x0, self.y0), (self.x0 + self.w, self.y0 + self.h), DIM, 1)
        whole = all(float(v).is_integer() for v in self.ticks)
        for v in self.ticks:
            cv2.line(img, (self.x0, self.Y(v)), (self.x0 + self.w, self.Y(v)), GRID, 1)
            label = f"{v:+.0f}" if whole else f"{v:+.2f}" if self.hi - self.lo < 4 else f"{v:+.1f}"
            text(img, label, (self.x0 - 58, self.Y(v) + 5), 0.42, GREY)
        for km in np.arange(0, self.rmax + 1e-6, 0.5):
            text(img, f"{km:g}", (self.X(km * 1000) - 8, self.y0 + self.h + 18), 0.42, GREY)
        text(img, "range to threshold (km)", (self.x0 + self.w - 190, self.y0 + self.h + 38), 0.42, GREY)
        for pts, color in series:
            if len(pts) > 1:
                cv2.polylines(img, [np.array([(self.X(r), self.Y(v)) for r, v in pts], np.int32)],
                              False, color, 2, cv2.LINE_AA)


class Renderer:
    def __init__(self, result, estimator):
        self.r, self.est = result, estimator
        recs = result.records
        pad = lambda vals, m: (min(vals) - m, max(vals) + m)
        xt = [v for r in recs for v in (r["xt_gt"], r["xt_est"]) if np.isfinite(v)]
        yw = [v for r in recs for v in (r["yaw_gt"], r["yaw_est"]) if np.isfinite(v)]
        rmax = max(0.5, np.ceil(max(r["range_m"] for r in recs) / 500) * 0.5)
        # Cross-track grid at whole metres, at least 1 m apart (2 m past an 8 m span), so a
        # few tens of centimetres of error does not fill the plot.
        lo, hi = np.floor(min(xt) - 0.25), np.ceil(max(xt) + 0.25)
        if hi - lo < 2:
            lo, hi = lo - 1, hi + 1
        step = 1.0 if hi - lo <= 8 else 2.0
        self.p_xt = Plot(CAM_W + 80, 60, 540, 380, "cross-track, + right (m)", lo, hi, rmax,
                         ticks=np.arange(lo, hi + 1e-9, step))
        self.p_yaw = Plot(CAM_W + 80, 540, 540, 380, "yaw relative to runway (deg)", *pad(yw, 0.1), rmax)
        self.pos = {r["frame"]: k for k, r in enumerate(recs)}

    def frame(self, index) -> np.ndarray:
        """The replay frame for one frame index (BGR, 1920x1080)."""
        f, recs = self.r.flight, self.r.records
        k = self.pos[index]
        rec = recs[k]
        img = f.image(index).astype(np.float32)
        m = (self.r.masks[index].astype(np.float32) / 255)[..., None]
        img = (img * (1 - 0.45 * m) + np.array(TEAL, np.float32) * 0.45 * m).astype(np.uint8)
        gt = f.label_mask(index)
        if gt is not None:
            cs, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.polylines(img, cs, True, WHITE, 2, cv2.LINE_AA)
        if np.isfinite(rec["yaw_est"]):
            q = self.est.outline(f, self.r.calibration, index, rec["yaw_est"], -rec["xt_est"])
            if q is not None:
                cv2.polylines(img, [q], True, AMBER, 2, cv2.LINE_AA)

        canvas = np.full((H, W, 3), BG, np.uint8)
        canvas[:CAM_H, :CAM_W] = img
        done = recs[:k + 1]
        for plot, key in ((self.p_xt, "xt"), (self.p_yaw, "yaw")):
            plot.draw(canvas, [([(r["range_m"], r[f"{key}_gt"]) for r in done], WHITE),
                               ([(r["range_m"], r[f"{key}_est"]) for r in done if np.isfinite(r[f"{key}_est"])], AMBER)])

        y0 = CAM_H + 50
        text(canvas, f"{f.name}   {f.airport} {f.runway}   DeepLabV3+ masks, crab-angle EKF", (30, y0), 0.75, INK, 2)
        text(canvas, f"weather: {f.weather}   visibility {f.visibility_m / 1000:.2f} km   range {rec['range_m']:,.0f} m"
                     f"   height {rec['height_m']:.0f} m   mask IoU {rec['iou']:.3f}", (30, y0 + 38), 0.62, GREY)
        rows = [("truth", WHITE, rec["xt_gt"], rec["yaw_gt"], None),
                ("estimate", AMBER, rec["xt_est"], rec["yaw_est"], (rec["xt_err"], rec["yaw_err"]))]
        for n, (name, color, xt, yw, err) in enumerate(rows):
            yy = y0 + 90 + 38 * n
            cv2.line(canvas, (30, yy - 6), (70, yy - 6), color, 3)
            text(canvas, name, (82, yy), 0.65)
            text(canvas, f"cross-track {xt:+6.2f} m", (230, yy), 0.65)
            text(canvas, f"yaw {yw:+.3f} deg", (500, yy), 0.65)
            if err is not None:
                text(canvas, f"error {err[0]:+5.2f} m / {err[1]:+.3f} deg", (740, yy), 0.65)
        text(canvas, "camera: teal = DeepLab mask, white = label, amber = runway drawn from the estimate",
             (30, H - 20), 0.5, GREY)
        return canvas


def write_video(result, estimator, path, fps=15) -> None:
    rend = Renderer(result, estimator)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                           "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
                           "-pix_fmt", "yuv420p", "-crf", "22", str(path)], stdin=subprocess.PIPE)
    for r in result.records:
        ff.stdin.write(rend.frame(r["frame"]).tobytes())
    ff.stdin.close()
    if ff.wait():
        raise RuntimeError(f"ffmpeg failed writing {path}")
