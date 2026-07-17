# Copyright (c) OpenMMLab. All rights reserved.
"""Video capture helpers: single V4L2/file capture and a synchronized
stereo pair."""
import cv2


def open_capture(cam_cfg):
    src = str(cam_cfg.get('path', '/dev/video0'))
    cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
    if not cap.isOpened():
        raise RuntimeError(
            f'Could not open video source {src!r}. If it is a camera, make '
            'sure the device is passed into the container (see bootstrap.sh) '
            'and not in use by another application.')
    # Force the pixel format BEFORE size/fps: many UVC cameras (e.g. Arducam
    # OV9782) only reach their full frame rate in MJPG, while OpenCV's V4L2
    # backend defaults to YUYV, which may be capped as low as 10 FPS.
    fourcc = str(cam_cfg.get('fourcc', 'MJPG'))
    if fourcc and len(fourcc) == 4:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    if cam_cfg.get('width'):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(cam_cfg['width']))
    if cam_cfg.get('height'):
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cam_cfg['height']))
    if cam_cfg.get('fps'):
        cap.set(cv2.CAP_PROP_FPS, float(cam_cfg['fps']))
    # keep at most one buffered frame so the stream stays live (low latency)
    # even when inference is slower than the camera frame rate
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f'Capture negotiated for {src}: {w:.0f}x{h:.0f} @ '
          f'{cap.get(cv2.CAP_PROP_FPS):.0f} FPS')
    return cap


class StereoPair:
    """Two captures read as a pair.

    Both frames are grab()bed back-to-back before either is decoded
    (retrieve()), which bounds the time skew between the two views to a
    couple of milliseconds without threads or hardware sync.
    """

    def __init__(self, cam_cfg0, cam_cfg1):
        self.cap0 = open_capture(cam_cfg0)
        self.cap1 = open_capture(cam_cfg1)

    def read(self):
        """Returns (ok, frame0, frame1)."""
        if not self.cap0.grab() or not self.cap1.grab():
            return False, None, None
        ok0, f0 = self.cap0.retrieve()
        ok1, f1 = self.cap1.retrieve()
        if not (ok0 and ok1):
            return False, None, None
        return True, f0, f1

    def release(self):
        self.cap0.release()
        self.cap1.release()
