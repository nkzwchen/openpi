# ruff: noqa: PLC0415
"""Offline expert-interval review of a trusted absolute-six recording; never opens hardware."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from openpi.so101.contract import CAMERAS
from openpi.so101.contract import FPS
from openpi.so101.export import load_recording
from openpi.so101.export import metadata_error
from openpi.so101.export import review_template
from openpi.so101.export import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--reviewer", required=True)
    args = parser.parse_args()
    if not args.reviewer.strip():
        parser.error("Nonempty reviewer required")
    payload = load_recording(args.recording)
    if error := metadata_error(payload["meta"]):
        parser.error(f"Recording contract rejected: {error}")
    steps = payload["transitions"]
    if not steps:
        parser.error("Recording has no transitions")
    template = review_template(args.recording)
    destination = args.recording.with_suffix(".expert.json")
    intervals = []
    if destination.exists():
        previous = json.loads(destination.read_text())
        if previous.get("recording_sha256") != template["recording_sha256"]:
            parser.error("Existing review hash mismatch")
        intervals.extend(previous.get("intervals", []))
    import tkinter as tk
    from tkinter import messagebox

    from PIL import Image
    from PIL import ImageOps
    from PIL import ImageTk

    root = tk.Tk()
    root.title("OpenPI expert review — full images / absolute6")
    frame = tk.Frame(root)
    frame.pack()
    panes = [tk.Label(frame) for _ in CAMERAS]
    for column, pane in enumerate(panes):
        pane.grid(row=0, column=column)
    detail = tk.StringVar()
    tk.Label(root, textvariable=detail, justify="left").pack()
    current = tk.IntVar(value=0)
    start = [None]
    playing = [False]
    images = []
    selected = tk.Listbox(root, width=80, height=5)
    selected.pack()

    def refresh_intervals():
        selected.delete(0, tk.END)
        for interval in intervals:
            selected.insert(tk.END, f"[{interval['start']}, {interval['end']})")

    def show(_=None):
        images.clear()
        step = steps[current.get()]
        info = step["infos"]
        for camera, pane in zip(CAMERAS, panes, strict=True):
            im = Image.fromarray(np.asarray(step["observations"][camera]).squeeze())
            images.append(ImageTk.PhotoImage(ImageOps.contain(im, (480, 360))))
            pane.configure(image=images[-1])
        detail.set(
            f"Original transition {current.get()}/{len(steps) - 1} | human={info.get('intervening')} | quality={info.get('quality_ok')}\n"
            f"state={np.asarray(step['observations']['state']).round(2)}\naction={np.asarray(step['actions']).round(2)}\n"
            f"segment_start={info.get('segment_start')} | recording_step={info.get('recording_step')} | interval start={start[0]}"
        )

    tk.Scale(root, from_=0, to=len(steps) - 1, orient="horizontal", variable=current, command=show, length=960).pack()
    buttons = tk.Frame(root)
    buttons.pack()

    def move(delta):
        current.set(max(0, min(len(steps) - 1, current.get() + delta)))
        show()

    def mark_start():
        start[0] = current.get()
        show()

    def mark_end():
        if start[0] is None or current.get() < start[0]:
            messagebox.showerror("Interval", "Mark a start at or before the current frame")
            return
        interval = {"start": start[0], "end": current.get() + 1}
        if any(max(interval["start"], i["start"]) < min(interval["end"], i["end"]) for i in intervals):
            messagebox.showerror("Interval", "Intervals must not overlap")
            return
        intervals.append(interval)
        intervals.sort(key=lambda i: i["start"])
        start[0] = None
        refresh_intervals()
        show()

    def remove():
        if selected.curselection():
            intervals.pop(selected.curselection()[0])
            refresh_intervals()

    def play():
        playing[0] = not playing[0]

    def timer():
        if playing[0]:
            move(1)
        root.after(round(1000 / FPS), timer)

    def save(status):
        if sha256(args.recording) != template["recording_sha256"]:
            messagebox.showerror("Hash", "Source recording changed; reopen review")
            return
        review = {
            **template,
            "reviewer": args.reviewer,
            "reviewed": True,
            "status": status,
            "intervals": intervals if status == "approved" else [],
            "reviewed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if destination.exists():
            backup = destination.with_name(destination.name + f".{time.time_ns()}.bak")
            backup.write_bytes(destination.read_bytes())
        temporary = destination.with_suffix(".json.partial")
        temporary.write_text(json.dumps(review, indent=2))
        temporary.replace(destination)
        root.destroy()

    for label, command in [
        ("Previous", lambda: move(-1)),
        ("Next", lambda: move(1)),
        ("Play/Pause", play),
        ("Start interval", mark_start),
        ("End (include current)", mark_end),
        ("Remove interval", remove),
        ("Approve selected intervals", lambda: save("approved")),
        ("Reject whole recording", lambda: save("rejected")),
    ]:
        tk.Button(buttons, text=label, command=command).pack(side="left")
    root.bind("<Left>", lambda _: move(-1))
    root.bind("<Right>", lambda _: move(1))
    tk.Label(
        root,
        text="Only reviewed human actions are exported. Policy actions, bad quality, pauses and clipped commands are filtered separately.",
    ).pack()
    refresh_intervals()
    show()
    timer()
    root.mainloop()


if __name__ == "__main__":
    main()
