import os
import shutil

def merge_data():
    # Folders to merge
    data_dirs = ["data/N1", "data/N2", "data/N3"]
    output_dir = "data/merged"
    output_img_dir = os.path.join(output_dir, "img")
    os.makedirs(output_img_dir, exist_ok=True)
    
    output_captions_path = os.path.join(output_dir, "captions.txt")
    with open(output_captions_path, "w", encoding="utf-8") as out_captions:
        for d in data_dirs:
            img_dir = os.path.join(d, "img")
            captions_file = os.path.join(d, "captions.txt")
            if os.path.exists(img_dir):
                for fname in os.listdir(img_dir):
                    src_path = os.path.join(img_dir, fname)
                    if os.path.isfile(src_path):
                        dst_path = os.path.join(output_img_dir, fname)
                        shutil.copy2(src_path, dst_path)
            if os.path.exists(captions_file):
                with open(captions_file, "r", encoding="utf-8") as cap_file:
                    for line in cap_file:
                        out_captions.write(line)
    print("Merge completed.")

if __name__ == "__main__":
    merge_data()
