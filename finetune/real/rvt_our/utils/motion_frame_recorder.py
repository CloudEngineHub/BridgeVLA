"""
Motion-segment frame recorder for real-robot evaluation.

The policy predicts discrete keyframes: at each decision step the low-level
executor drives the arm to the next keypose. This recorder saves camera
frames ONLY while that physical motion happens (bot.point_control /
claws_control), at the ZED's native frame rate (HD1080 @ 30 fps). Nothing
is saved while the client waits for inference or user confirmation.

Output layout — same per-run directory as the Flask view logs
(``<ckpt dir>/eval/<model_stem>/ep<NNN>_<mmdd_HHMMSS>_<instruction_slug>/``,
from POST /reset ``session_dir``; renamed to ``…_success`` / ``…_fail`` /
``…_aborted`` when the episode ends)::

    <session_dir>/
        episode_meta.txt              (written by the server on /reset)
        step_0000/
            meta.txt                  (server: predicted pose for this step)
            views/ ...                (server: rendered view PNGs)
            frames/                   ← everything below is written HERE
                frame_000000.jpg ...  motion frames, camera-native ~30 fps
                timestamps.txt        frame_idx  camera_ts_ms  monotonic_s
                step_meta.txt         target pose + frame statistics
        step_0001/
            ...

Server and recorder share the step folder but never the same subfolder:
``views/`` is the server's, ``frames/`` is ours. STEP_DIR_FMT / FRAMES_SUBDIR
below mirror the constants in ``eval_flask_app.py`` — keep them in sync.

Stitch with::

    python exec_frames_to_video.py \
        <ckpt dir>/eval/model_1500/0718_190656_swap-the-two-eggplants_success

Threading contract: the ZED SDK does not support concurrent access to one
camera from several threads. Between start() and stop() the capture thread
is the ONLY user of the camera, and stop() joins it before returning — so
callers must stop() before the main loop's next Camera.capture(). JPEG
encoding runs on writer threads behind a bounded queue, so the capture loop
never stalls on disk I/O.
"""

import os
import time
import queue
import threading

import cv2

# Layout contract with eval_flask_app.py (see module docstring).
STEP_DIR_FMT  = "step_{:04d}"
FRAMES_SUBDIR = "frames"


class MotionFrameRecorder:
    """Records RGB frames from an already-open ZedCam during motion segments."""

    def __init__(self, zed_cam, episode_dir, jpeg_quality=95, n_writers=2,
                 max_queue=256, resize_wh=None):
        """
        Args:
            zed_cam:      ``utils.real_camera_utils.ZedCam`` instance
                          (the raw single camera, NOT the ``Camera`` wrapper).
            episode_dir:  eval session dir (shared with Flask view logs);
                          step_XXXX/frames/ folders are created under it.
            jpeg_quality: cv2 JPEG quality (95 ≈ visually lossless).
            n_writers:    JPEG writer threads (2 keeps up with 1080p@30fps).
            max_queue:    max in-flight frames before new ones are dropped
                          (drops are counted and reported, never silent).
            resize_wh:    optional (width, height) to downscale saved frames;
                          None keeps camera-native resolution.
        """
        self.cam = zed_cam
        self.episode_dir = episode_dir
        self.jpeg_quality = int(jpeg_quality)
        self.n_writers = int(n_writers)
        self.max_queue = int(max_queue)
        self.resize_wh = resize_wh

        self._active = False
        self._stop_evt = threading.Event()
        self._queue = None
        self._capture_thread = None
        self._writer_threads = []
        self._step_dir = None
        self._pending_meta = []
        self._timestamps = []   # (frame_idx, camera_ts_ms, monotonic_s)
        self._dropped = 0
        self._t_start = None

        os.makedirs(self.episode_dir, exist_ok=True)

    # lifecycle
    @property
    def last_step_dir(self):
        """Frames folder of the most recent step (``step_XXXX/frames/``)."""
        return self._step_dir

    def start(self, step_idx, meta_lines=None):
        """Begin recording one step's motion segment.

        Call immediately before the blocking move command and pair with
        stop() right after it returns (use try/finally so an errored move
        still stops the capture thread).
        """
        if self._active:
            raise RuntimeError("MotionFrameRecorder.start() while already recording")

        self._step_dir = os.path.join(
            self.episode_dir, STEP_DIR_FMT.format(step_idx), FRAMES_SUBDIR)
        os.makedirs(self._step_dir, exist_ok=True)
        self._pending_meta = list(meta_lines) if meta_lines else []

        self._stop_evt.clear()
        self._queue = queue.Queue(maxsize=self.max_queue)
        self._timestamps = []
        self._dropped = 0
        self._t_start = time.monotonic()

        self._writer_threads = [
            threading.Thread(target=self._writer_loop, daemon=True,
                             name=f"exec-frame-writer-{i}")
            for i in range(self.n_writers)
        ]
        for t in self._writer_threads:
            t.start()
        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="exec-frame-capture")
        self._capture_thread.start()
        self._active = True

    def stop(self):
        """Stop recording, flush queued frames to disk, write metadata.

        Returns:
            (n_frames, measured_fps). Safe to call when not recording.
        """
        if not self._active:
            return 0, 0.0

        self._stop_evt.set()
        self._capture_thread.join()
        for _ in self._writer_threads:
            self._queue.put(None)
        for t in self._writer_threads:
            t.join()
        self._active = False

        duration_s = time.monotonic() - self._t_start
        n = len(self._timestamps)
        # Prefer camera timestamps for the fps estimate; fall back to wall time.
        if n >= 2:
            cam_span_s = (self._timestamps[-1][1] - self._timestamps[0][1]) / 1000.0
            fps = (n - 1) / cam_span_s if cam_span_s > 0 else n / max(duration_s, 1e-6)
        else:
            fps = 0.0

        with open(os.path.join(self._step_dir, "timestamps.txt"), "w") as f:
            f.write("# frame_idx  camera_ts_ms  monotonic_s\n")
            for idx, cam_ts_ms, mono_s in self._timestamps:
                f.write(f"{idx:06d}  {cam_ts_ms:.3f}  {mono_s:.6f}\n")

        meta = list(self._pending_meta) + [
            f"n_frames: {n}",
            f"dropped_frames: {self._dropped}",
            f"duration_s: {duration_s:.3f}",
            f"measured_fps: {fps:.2f}",
        ]
        with open(os.path.join(self._step_dir, "step_meta.txt"), "w") as f:
            f.write("\n".join(meta) + "\n")

        if self._dropped:
            print(f"[MotionFrameRecorder] WARNING: dropped {self._dropped} "
                  f"frames in {self._step_dir} (writer queue full)")
        return n, fps

    def close(self):
        """Idempotent shutdown; call before ZedCam.stop() / Camera.stop()."""
        try:
            self.stop()
        except Exception as e:
            print(f"[MotionFrameRecorder] close() error ignored: {e}")

    # worker threads
    def _capture_loop(self):
        frame_idx = 0
        while not self._stop_evt.is_set():
            frame, cam_ts_ms = self.cam.capture_rgb()
            if frame is None:
                time.sleep(0.005)   # transient grab failure — avoid busy spin
                continue
            mono_s = time.monotonic()
            try:
                self._queue.put_nowait((frame_idx, frame))
            except queue.Full:
                self._dropped += 1
                continue
            self._timestamps.append((frame_idx, cam_ts_ms, mono_s))
            frame_idx += 1

    def _writer_loop(self):
        while True:
            item = self._queue.get()
            if item is None:
                return
            frame_idx, frame = item
            if self.resize_wh is not None:
                frame = cv2.resize(frame, tuple(self.resize_wh),
                                   interpolation=cv2.INTER_AREA)
            cv2.imwrite(
                os.path.join(self._step_dir, f"frame_{frame_idx:06d}.jpg"),
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
            )
