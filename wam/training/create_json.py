import h5py
import cv2
import numpy as np
import json
import argparse
import sys

def main(hdf5_path, output_json):
    print(f"Loading {hdf5_path}...")
    try:
        f = h5py.File(hdf5_path, 'r')
    except Exception as e:
        print(f"Failed to open HDF5 file: {e}")
        sys.exit(1)

    front_img = f['front_image']
    wrist_img = f['wrist_image']
    num_frames = front_img.shape[0]

    # State variables
    frame_idx = 0
    last_frame_idx = -1
    start_idx = -1
    end_idx = -1
    segments = []
    playing = False

    # Create UI
    window_name = 'Episode Splitter (Press H for Help)'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def on_trackbar(val):
        nonlocal frame_idx
        frame_idx = val

    cv2.createTrackbar('Frame', window_name, 0, num_frames - 1, on_trackbar)

    print("\nControls:")
    print("  [Space]   : Play / Pause")
    print("  [A] / [D] : Step backward / forward one frame")
    print("  [S]       : Mark Start frame")
    print("  [E]       : Mark End frame")
    print("  [C]       : Commit segment (prompts for task name in terminal)")
    print("  [Q]       : Save and Quit\n")

    while True:
        if playing:
            frame_idx = min(frame_idx + 1, num_frames - 1)
            cv2.setTrackbarPos('Frame', window_name, frame_idx)

        # Only read from disk if the frame changed (keeps the UI responsive)
        if frame_idx != last_frame_idx:
            img_f = front_img[frame_idx]
            img_w = wrist_img[frame_idx]
            
            # HDF5 images are usually RGB, OpenCV expects BGR
            img_f = cv2.cvtColor(img_f, cv2.COLOR_RGB2BGR)
            img_w = cv2.cvtColor(img_w, cv2.COLOR_RGB2BGR)
            
            combined = cv2.hconcat([img_f, img_w])
            last_frame_idx = frame_idx

        # Create a copy for drawing overlays
        display = combined.copy()

        # UI Text Overlays
        cv2.putText(display, f"Frame: {frame_idx} / {num_frames-1}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, f"Start: {start_idx if start_idx != -1 else 'Not set'}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)
        cv2.putText(display, f"End: {end_idx if end_idx != -1 else 'Not set'}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(display, f"Saved Segments: {len(segments)}", (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        display = cv2.resize(display, (1920, 1080))

        cv2.imshow(window_name, display)

        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            playing = not playing
        elif key == ord('a') or key == 81: # 'a' or Left Arrow
            frame_idx = max(0, frame_idx - 1)
            cv2.setTrackbarPos('Frame', window_name, frame_idx)
        elif key == ord('d') or key == 83: # 'd' or Right Arrow
            frame_idx = min(num_frames - 1, frame_idx + 1)
            cv2.setTrackbarPos('Frame', window_name, frame_idx)
        elif key == ord('s'):
            start_idx = frame_idx
            print(f"[*] Start marked at frame {start_idx}")
        elif key == ord('e'):
            end_idx = frame_idx
            print(f"[*] End marked at frame {end_idx}")
        elif key == ord('c'):
            if start_idx != -1 and end_idx != -1 and start_idx <= end_idx:
                playing = False # Pause video to accept input
                task_name = input(f"\n>>> Enter task name for segment [{start_idx} : {end_idx}]: ")
                
                segments.append({
                    "start": start_idx,
                    "end": end_idx,
                    "task_name": task_name.strip()
                })
                print(f"[+] Saved segment: {segments[-1]}\n")
                
                # Reset for next annotation
                start_idx = -1
                end_idx = -1
            else:
                print("[!] Invalid operation. Ensure Start and End are set, and Start <= End.")

    f.close()
    cv2.destroyAllWindows()

    print("\n==================================")
    print("Final Annotations:")
    print(json.dumps(segments, indent=2))
    
    with open(output_json, 'w') as out_f:
        json.dump(segments, out_f, indent=2)
    print(f"\nSaved all splits to {output_json}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Annotate start/end frames for HDF5 robotic episodes.")
    parser.add_argument("hdf5_path", help="Path to the HDF5 file.")
    parser.add_argument("--out", default="episode_splits.json", help="Output JSON file name.")
    args = parser.parse_args()
    main(args.hdf5_path, args.out)
