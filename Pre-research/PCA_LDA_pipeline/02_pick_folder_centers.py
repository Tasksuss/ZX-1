import os
import cv2
import json
import argparse

clicked_point = None


def mouse_callback(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_path", required=True)
    args = parser.parse_args()

    with open(args.metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    for folder_name, info in metadata.items():
        ref_path = info["reference_path"]
        if not os.path.exists(ref_path):
            print(f"[WARN] missing reference: {ref_path}")
            continue

        img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARN] cannot read: {ref_path}")
            continue

        disp = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        current = info.get("center_x"), info.get("center_y")
        if current[0] is not None and current[1] is not None:
            cv2.circle(disp, (int(current[0]), int(current[1])), 5, (0, 0, 255), -1)

        global clicked_point
        clicked_point = None

        win = f"{folder_name} - click throat center, press S to save, N to skip, Q to quit"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win, mouse_callback)

        while True:
            show = disp.copy()
            if clicked_point is not None:
                cv2.circle(show, clicked_point, 5, (0, 255, 0), -1)

            cv2.imshow(win, show)
            key = cv2.waitKey(30) & 0xFF

            if key == ord("s"):
                if clicked_point is not None:
                    metadata[folder_name]["center_x"] = int(clicked_point[0])
                    metadata[folder_name]["center_y"] = int(clicked_point[1])
                    print(f"[SAVE] {folder_name}: {clicked_point}")
                break
            elif key == ord("n"):
                print(f"[SKIP] {folder_name}")
                break
            elif key == ord("q"):
                cv2.destroyAllWindows()
                with open(args.metadata_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2)
                print("[DONE] early quit, metadata saved")
                return

        cv2.destroyWindow(win)

    with open(args.metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[DONE] updated metadata: {args.metadata_path}")


if __name__ == "__main__":
    main()




"""
python pick_folder_centers.py \
  --metadata_path /Users/apple/Downloads/Troat/workspace/folder_metadata.json
"""