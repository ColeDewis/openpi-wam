# uv run python3 -m wam.training.create_json ./dataset
import h5py
import cv2
import numpy as np
import json
import argparse
import sys
from pathlib import Path

def main(dataset_dir, output_json):
    dataset_path = Path(dataset_dir)
    files = sorted([f for f in dataset_path.iterdir() if f.suffix == '.hdf5'])
    
    if not files:
        sys.exit(1)

    results = {f.name: [] for f in files}
    file_idx, frame_idx = 0, 0
    playing = False
    f, front_img, wrist_img, num_frames = None, None, None, 0

    window_name = 'Episode Annotator'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def on_trackbar(val):
        nonlocal frame_idx
        frame_idx = val

    cv2.createTrackbar('Frame', window_name, 0, 100, on_trackbar)

    while True:
        current_file = files[file_idx]

        if f is None:
            f = h5py.File(current_file, 'r')
            front_img, wrist_img = f['front_image'], f['wrist_image']
            num_frames = front_img.shape[0]
            frame_idx, last_frame_idx = 0, -1
            
            cv2.setTrackbarMax('Frame', window_name, max(1, num_frames - 1))
            cv2.setTrackbarPos('Frame', window_name, 0)

        if playing:
            frame_idx = min(frame_idx + 1, num_frames - 1)
            cv2.setTrackbarPos('Frame', window_name, frame_idx)

        if frame_idx != last_frame_idx:
            img_f = cv2.cvtColor(front_img[frame_idx], cv2.COLOR_RGB2BGR)
            img_w = cv2.cvtColor(wrist_img[frame_idx], cv2.COLOR_RGB2BGR)
            combined = cv2.hconcat([img_f, img_w])
            last_frame_idx = frame_idx

        display = combined.copy()
        
        cv2.putText(display, f"File {file_idx+1}/{len(files)}: {current_file.name}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, f"Frame: {frame_idx}/{num_frames-1}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, f"Tasks: {', '.join(results[current_file.name])}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        display = cv2.resize(display, (800, 450))
        cv2.imshow(window_name, display)

        key = cv2.waitKey(100) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            playing = not playing
        elif key in [ord('a'), 81]: 
            frame_idx = max(0, frame_idx - 1)
            cv2.setTrackbarPos('Frame', window_name, frame_idx)
        elif key in [ord('d'), 83]: 
            frame_idx = min(num_frames - 1, frame_idx + 1)
            cv2.setTrackbarPos('Frame', window_name, frame_idx)
        elif key == ord('n'):
            f.close()
            f = None
            file_idx = min(len(files) - 1, file_idx + 1)
        elif key == ord('p'):
            f.close()
            f = None
            file_idx = max(0, file_idx - 1)
        elif key == ord('c'):
            playing = False
            task = input(f"\n>>> Enter task name for {current_file.name}: ").strip()
            if task:
                results[current_file.name].append(task)

    if f is not None:
        f.close()
    cv2.destroyAllWindows()

    output_data = [{"file": fname, "task_names": tasks} for fname, tasks in results.items() if tasks]
    
    with open(output_json, 'w') as out_f:
        json.dump(output_data, out_f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir")
    parser.add_argument("--out", default="episode_splits.json")
    args = parser.parse_args()
    main(args.dataset_dir, args.out)
