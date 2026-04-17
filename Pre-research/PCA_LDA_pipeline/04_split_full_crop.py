import os
import shutil
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--mode", choices=["copy", "move"], default="copy")
    args = parser.parse_args()

    full_root = os.path.join(args.output_root, "full")
    crop_root = os.path.join(args.output_root, "crop")

    for root, _, files in os.walk(args.input_root):
        for file in files:
            if not file.endswith(".mp4"):
                continue

            src_path = os.path.join(root, file)

            # 相对路径（保持层级结构）
            rel_dir = os.path.relpath(root, args.input_root)

            # 判断类型
            if "_full" in file:
                dst_base = full_root
            elif "_crop" in file:
                dst_base = crop_root
            else:
                continue  # 跳过异常文件

            dst_dir = os.path.join(dst_base, rel_dir)
            os.makedirs(dst_dir, exist_ok=True)

            dst_path = os.path.join(dst_dir, file)

            if args.mode == "copy":
                shutil.copy2(src_path, dst_path)
            else:
                shutil.move(src_path, dst_path)

            print(f"[OK] {file} -> {dst_base}")


if __name__ == "__main__":
    main()


"""
python split_full_crop.py \
  --input_root /Users/apple/Downloads/Troat/workspace/processed \
  --output_root /Users/apple/Downloads/Troat/workspace/processed_split \
  --mode copy
"""