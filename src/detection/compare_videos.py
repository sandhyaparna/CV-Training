import cv2
import time
from pathlib import Path

def play_top_bottom(
    top_path,
    bottom_path,
    window_title="Top-Bottom",
    save_path=None,           # e.g., "out.avi" or "out.mp4"
    codec="XVID",             # "XVID"/"MJPG" for .avi; "mp4v"/"avc1" for .mp4
    output_fps=None,          # None -> min(input FPS)
    display=False             # Auto-disables gracefully if HighGUI is unavailable
):
    top_path, bottom_path = map(lambda p: str(Path(p)), (top_path, bottom_path))
    cap_t, cap_b = cv2.VideoCapture(top_path), cv2.VideoCapture(bottom_path)
    if not cap_t.isOpened() or not cap_b.isOpened():
        raise RuntimeError("Could not open one of the videos.")

    def _fps(cap, fallback=30.0):
        fps = cap.get(cv2.CAP_PROP_FPS)
        return fps if fps and fps > 1e-2 else fallback

    in_fps_t = _fps(cap_t)
    in_fps_b = _fps(cap_b)
    fps = float(output_fps) if output_fps else min(in_fps_t, in_fps_b)
    frame_dt = 1.0 / max(1e-6, fps)

    ok_t, frame_t = cap_t.read()
    ok_b, frame_b = cap_b.read()
    if not (ok_t and ok_b):
        cap_t.release(); cap_b.release()
        raise RuntimeError("Could not read frames from the videos.")

    # ---------- helpers for vertical stack (match widths) ----------
    def resize_to_width(img, target_w):
        h_i, w_i = img.shape[:2]
        if w_i == target_w:
            return img
        scale = target_w / float(w_i)
        new_h = int(round(h_i * scale))
        return cv2.resize(img, (target_w, new_h), interpolation=cv2.INTER_AREA)

    def pad_to_width(img, W):
        h_i, w_i = img.shape[:2]
        if w_i == W: return img
        pad = W - w_i
        left = pad // 2; right = pad - left
        return cv2.copyMakeBorder(img, 0, 0, left, right, cv2.BORDER_CONSTANT, value=(0,0,0))

    # ---------- choose fixed composition geometry ----------
    target_w = frame_t.shape[1]  # match bottom to top's initial width
    frame_b = resize_to_width(frame_b, target_w)

    top0    = pad_to_width(frame_t, target_w)
    bottom0 = pad_to_width(frame_b, target_w)
    combined0 = cv2.vconcat([top0, bottom0])

    # ---------- writer (optional) ----------
    writer = None
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out_size = (combined0.shape[1], combined0.shape[0])  # (width, height)
        writer = cv2.VideoWriter(str(save_path), fourcc, fps, out_size)
        if not writer.isOpened():
            cap_t.release(); cap_b.release()
            raise RuntimeError("Could not open VideoWriter. Try a different codec/container.")

    # ---------- try window (optional) ----------
    window_ready = False
    if display:
        try:
            cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
            window_ready = True
        except Exception:
            window_ready = False

    # ---------- timing / first frame ----------
    t0 = time.perf_counter()
    frame_idx = 0
    if writer: writer.write(combined0)
    if window_ready:
        cv2.imshow(window_title, combined0); cv2.waitKey(1)

    # ---------- loop ----------
    while True:
        ok_t, frame_t = cap_t.read()
        ok_b, frame_b = cap_b.read()
        if not ok_t or not ok_b:
            break

        b_resized = resize_to_width(frame_b, target_w)
        top    = pad_to_width(frame_t, target_w)
        bottom = pad_to_width(b_resized, target_w)
        combined = cv2.vconcat([top, bottom])

        if writer: writer.write(combined)
        if window_ready:
            cv2.imshow(window_title, combined); cv2.waitKey(1)

        # pace to target FPS (works headless too)
        frame_idx += 1
        target_time = t0 + frame_idx * frame_dt
        now = time.perf_counter()
        if target_time > now:
            time.sleep(target_time - now)

    # ---------- cleanup ----------
    cap_t.release(); cap_b.release()
    if writer: writer.release()
    if window_ready: cv2.destroyAllWindows()



def play_side_by_side(
    left_path,
    right_path,
    window_title="Side by Side",
    save_path=None,           # e.g., "out.avi" or "out.mp4"
    codec="XVID",             # "XVID" or "MJPG" for .avi; "mp4v"/"avc1" for .mp4
    output_fps=None,          # None -> min(input FPS)
    display=True              # Try to show a window; will auto-disable if headless
):
    left_path, right_path = map(lambda p: str(Path(p)), (left_path, right_path))
    cap_l, cap_r = cv2.VideoCapture(left_path), cv2.VideoCapture(right_path)
    if not cap_l.isOpened() or not cap_r.isOpened():
        raise RuntimeError("Could not open one of the videos.")

    def _fps(cap, fallback=30.0):
        fps = cap.get(cv2.CAP_PROP_FPS)
        return fps if fps and fps > 1e-2 else fallback

    in_fps_l = _fps(cap_l)
    in_fps_r = _fps(cap_r)
    fps = float(output_fps) if output_fps else min(in_fps_l, in_fps_r)
    frame_dt = 1.0 / max(1e-6, fps)

    ok_l, frame_l = cap_l.read()
    ok_r, frame_r = cap_r.read()
    if not (ok_l and ok_r):
        cap_l.release(); cap_r.release()
        raise RuntimeError("Could not read frames from the videos.")

    def resize_to_height(img, target_h):
        h_i, w_i = img.shape[:2]
        if h_i == target_h:
            return img
        scale = target_h / float(h_i)
        new_w = int(round(w_i * scale))
        return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)

    target_h = frame_l.shape[0]
    frame_r = resize_to_height(frame_r, target_h)

    def pad_to_height(img, H):
        if img.shape[0] == H: return img
        pad = H - img.shape[0]
        top = pad // 2; bottom = pad - top
        return cv2.copyMakeBorder(img, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))

    left0  = pad_to_height(frame_l, target_h)
    right0 = pad_to_height(frame_r, target_h)
    combined0 = cv2.hconcat([left0, right0])

    # Prepare writer (optional)
    writer = None
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out_size = (combined0.shape[1], combined0.shape[0])
        writer = cv2.VideoWriter(str(save_path), fourcc, fps, out_size)
        if not writer.isOpened():
            cap_l.release(); cap_r.release()
            raise RuntimeError("Could not open VideoWriter. Try a different codec/container.")

    # Try to prepare a window; auto-disable display if HighGUI is unavailable
    window_ready = False
    if display:
        try:
            cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
            window_ready = True
        except Exception:
            # Headless fallback
            window_ready = False

    # Timing loop
    t0 = time.perf_counter()
    frame_idx = 0

    # Process the first frame too
    if writer: writer.write(combined0)
    if window_ready:
        cv2.imshow(window_title, combined0)
        cv2.waitKey(1)

    while True:
        ok_l, frame_l = cap_l.read()
        ok_r, frame_r = cap_r.read()
        if not ok_l or not ok_r:
            break

        frame_r_resized = resize_to_height(frame_r, target_h)
        left  = pad_to_height(frame_l, target_h)
        right = pad_to_height(frame_r_resized, target_h)
        combined = cv2.hconcat([left, right])

        if writer:
            writer.write(combined)

        if window_ready:
            cv2.imshow(window_title, combined)
            # Keep UI responsive; ~1ms is fine since we pace with sleep below
            cv2.waitKey(1)

        # Pace to target FPS even without GUI
        frame_idx += 1
        target_time = t0 + frame_idx * frame_dt
        now = time.perf_counter()
        if target_time > now:
            time.sleep(target_time - now)

    cap_l.release(); cap_r.release()
    if writer: writer.release()
    if window_ready:
        cv2.destroyAllWindows()


