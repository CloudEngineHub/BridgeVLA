"""
Stitch per-step motion frames recorded during real-robot eval into a video.

Input layout (produced by eval_client.py with
RECORD_EXEC_FRAMES=True; frames share the Flask session dir, which lives
next to the evaluated checkpoint)::

    <ckpt dir>/eval/<model_stem>/ep<NNN>_<mmdd_HHMMSS>_<instruction_slug>_<result>/
        step_0000/frames/frame_000000.jpg ...   (+ views/ PNGs alongside)
        step_0001/frames/frame_000000.jpg ...

Runs are found by content (a folder with ``step_*/frame_*``), not by name, so
both this layout and the older prefix-less ``<mmdd_HHMMSS>_<slug>_<result>``
are picked up.

Usage::

    # one eval run -> <session_dir>/episode_video.mp4
    python exec_frames_to_video.py \
        /path/to/7_16_real_memory_.../eval/model_1500/ep007_0718_190656_swap_success

    # every run with motion frames anywhere under a root (searched recursively,
    # so a checkpoint dir, its eval/ folder, or a whole logs tree all work)
    python exec_frames_to_video.py /path/to/7_16_real_memory_... --all

    # options
    python exec_frames_to_video.py <session_dir> \
        --fps 30            # output frame rate (recording is ~30 fps)
        --annotate          # draw the step index on each frame
        --pause-frames 15   # hold each step's last frame N frames (visual gap)
        --per-step          # additionally write one video per step
        --out my.mp4        # custom output path (single-episode mode only)

Only depends on OpenCV. Tries the requested codec first, then falls back to
mp4v, then XVID/.avi, so it works with a bare pip-installed opencv-python.
"""

import os
import glob
import argparse

import cv2


FRAMES_SUBDIR = "frames"   # matches utils/motion_frame_recorder.py


def _list_frames(step_dir):
    """Return sorted motion-frame paths for one step_*, or [].

    Looks in ``step_XXXX/frames/`` first, then falls back to ``step_XXXX/``
    itself so runs recorded before frames got their own subfolder still work.
    """
    for base in (os.path.join(step_dir, FRAMES_SUBDIR), step_dir):
        for ext in ("jpg", "png"):
            frames = sorted(glob.glob(os.path.join(base, f"frame_*.{ext}")))
            if frames:
                return frames
    return []


def collect_steps(episode_dir):
    """Return [(step_dir, [frame paths...]), ...] sorted by step then frame.

    Steps that only have view PNGs (no frame_*) are skipped silently — common
    when scanning a session that was not recorded with RECORD_EXEC_FRAMES.
    """
    step_dirs = sorted(
        d for d in glob.glob(os.path.join(episode_dir, "step_*"))
        if os.path.isdir(d)
    )
    steps = []
    for d in step_dirs:
        frames = _list_frames(d)
        if frames:
            steps.append((d, frames))
    return steps


def has_motion_frames(episode_dir):
    """True if any step_* under episode_dir contains frame_*.jpg/.png."""
    for d in glob.glob(os.path.join(episode_dir, "step_*")):
        if os.path.isdir(d) and _list_frames(d):
            return True
    return False


def find_episode_dirs(root):
    """Eval runs anywhere under *root* that have at least one motion-frame step.

    Walks recursively, so any level can serve as the root: a single run dir, an
    ``eval/<model_stem>/`` folder, a checkpoint dir, or a tree holding many
    training runs. Once a run matches, its ``step_*`` children are pruned from
    the walk, so no run is reported twice and the scan stays cheap.
    """
    found = []
    for dirpath, dirnames, _ in os.walk(root):
        if has_motion_frames(dirpath):
            found.append(dirpath)
            dirnames[:] = [d for d in dirnames if not d.startswith("step_")]
    return sorted(found)


def open_writer(out_path, codec, fps, size):
    """Open a cv2.VideoWriter, falling back through codecs if needed."""
    tried = []
    for c in [codec] + [x for x in ("mp4v", "XVID") if x != codec]:
        path = out_path
        if c == "XVID":
            path = os.path.splitext(out_path)[0] + ".avi"
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*c), fps, size)
        if vw.isOpened():
            if c != codec:
                print(f"[warn] codec '{codec}' unavailable, using '{c}'")
            return vw, path
        vw.release()
        tried.append(c)
    raise RuntimeError(f"could not open a VideoWriter (tried {tried})")


def annotate_frame(img, text):
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    x, y = 12, 14 + th
    cv2.rectangle(img, (x - 8, y - th - 10),
                  (x + tw + 8, y + baseline + 6), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (255, 255, 255), 2, cv2.LINE_AA)
    return img


def write_video(steps, out_path, fps, do_annotate=False, pause_frames=0,
                codec="mp4v"):
    """Concatenate all steps' frames into one video. Returns (path, n_frames)."""
    first = cv2.imread(steps[0][1][0])
    if first is None:
        raise RuntimeError(f"cannot read first frame {steps[0][1][0]}")
    h, w = first.shape[:2]
    vw, out_path = open_writer(out_path, codec, fps, (w, h))

    n_written = 0
    for step_dir, frames in steps:
        label = os.path.basename(step_dir)
        last = None
        for fp in frames:
            img = cv2.imread(fp)
            if img is None:
                print(f"[warn] unreadable frame {fp}, skipping")
                continue
            if (img.shape[1], img.shape[0]) != (w, h):
                img = cv2.resize(img, (w, h))
            if do_annotate:
                img = annotate_frame(img, label)
            vw.write(img)
            last = img
            n_written += 1
        # optional freeze on the step's last frame to visually separate steps
        for _ in range(pause_frames if last is not None else 0):
            vw.write(last)
            n_written += 1
    vw.release()
    return out_path, n_written


def process_episode(episode_dir, args):
    steps = collect_steps(episode_dir)
    if not steps:
        print(f"[warn] {episode_dir}: no step_*/frame_* motion frames, skipping")
        return

    out_path = args.out if args.out else os.path.join(episode_dir, "episode_video.mp4")
    path, n = write_video(steps, out_path, args.fps,
                          do_annotate=args.annotate,
                          pause_frames=args.pause_frames,
                          codec=args.codec)
    print(f"[ok] {path}: {len(steps)} steps, {n} frames, "
          f"{n / args.fps:.1f}s @ {args.fps} fps")

    if args.per_step:
        for step_dir, frames in steps:
            sp, sn = write_video([(step_dir, frames)],
                                 os.path.join(episode_dir,
                                              os.path.basename(step_dir) + ".mp4"),
                                 args.fps, do_annotate=args.annotate,
                                 pause_frames=0, codec=args.codec)
            print(f"     {sp}: {sn} frames")


def main():
    parser = argparse.ArgumentParser(
        description="Stitch recorded eval motion frames into a video.")
    parser.add_argument(
        "path",
        help="eval run dir (<ckpt dir>/eval/<model_stem>/"
             "<stamp>_<instruction>_<result>), or any parent of one with --all",
    )
    parser.add_argument("--all", action="store_true",
                        help="treat PATH as a root and process every eval run "
                             "found recursively under it that has motion frames")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="output video frame rate (default: 30)")
    parser.add_argument("--codec", default="mp4v",
                        help="fourcc codec (default: mp4v; falls back automatically)")
    parser.add_argument("--annotate", action="store_true",
                        help="draw the step label on each frame")
    parser.add_argument("--pause-frames", type=int, default=0,
                        help="freeze each step's last frame for N frames")
    parser.add_argument("--per-step", action="store_true",
                        help="also write one video per step")
    parser.add_argument("--out", default=None,
                        help="output path (single-episode mode only)")
    args = parser.parse_args()

    if args.all:
        if args.out:
            parser.error("--out cannot be combined with --all")
        episodes = find_episode_dirs(args.path)
        if not episodes:
            print(f"no session directories with motion frames under {args.path}")
            return
        print(f"found {len(episodes)} session(s) with motion frames under {args.path}")
        for ep in episodes:
            process_episode(ep, args)
    else:
        process_episode(args.path, args)


if __name__ == "__main__":
    main()
