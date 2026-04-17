#raw dataset is from 1-million-fake-faces
#resize to 256 x 256

import glob,os
from typing import List, Union
import random
from PIL import Image
from tqdm import tqdm
from pathlib import Path
data_path = str(Path(__file__).resolve().parent.parent)+ "/data/"

def list_files(root_dir: Union[str, Path], extensions: List[str] = None, as_str: bool = True) -> List:
    root_dir = Path(root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"folder is not exist: {root_dir}")

    if extensions is None:
        extensions = [".png", ".jpg"]
    extensions = [ext.lower() for ext in extensions]

    candidates = root_dir.rglob("*")
    files = [
        f if not as_str else str(f)
        for f in candidates
        if f.is_file() and f.suffix.lower() in extensions
    ]
    return files

def resize_keep_all_content(input_file, output_file, size=(256, 256)):
    input_file = Path(input_file)
    output_file = Path(output_file)

    try:
        img = Image.open(input_file)
        has_alpha = img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)
        scale = min(size[0] / img.width, size[1] / img.height)
        new_width = int(img.width * scale)
        new_height = int(img.height * scale)
        img_resized = img.resize((new_width, new_height), Image.LANCZOS)
        if has_alpha:
            background = Image.new("RGBA", size, (0, 0, 0, 0))
            offset_x = (size[0] - new_width) // 2
            offset_y = (size[1] - new_height) // 2
            background.paste(img_resized, (offset_x, offset_y), img_resized)
        else:
            bg_color = img.getpixel((0, 0))
            if isinstance(bg_color, int): 
                bg_color = (bg_color, bg_color, bg_color)
            elif len(bg_color) == 4:  # RGBA to RGB
                bg_color = bg_color[:3]

            background = Image.new("RGB", size, bg_color)
            offset_x = (size[0] - new_width) // 2
            offset_y = (size[1] - new_height) // 2
            background.paste(img_resized, (offset_x, offset_y))

        background.save(output_file, format="PNG")

    except Exception as e:
        print(f"failed: {input_file} — {e}")

def get_fake_faces():
    folder = f"{data_path}/raw/dataset/1-million-fake-faces"
    all_files = list_files(folder,['.jpg','.png'])
    return all_files

def get_real_faces():
    folder = f"{data_path}/raw/dataset/flickrfaceshq-dataset-ffhq"
    all_files = list_files(folder,['.jpg','.png'])
    return all_files

def main():
    fake_files = get_fake_faces()
    real_files = get_real_faces()
    min_count = min(len(fake_files),len(real_files))
    print(min_count,len(fake_files),len(real_files))
    for i in tqdm(range(min_count)):
        source = str(fake_files[i])
        file_name = source.split("/")[-1].split(".")[0]
        os.makedirs(f"{data_path}/face/resize/256/fake/", exist_ok=True)
        dest = f"{data_path}/face/resize/256/fake/{i}.png"
        resize_keep_all_content(source,dest)
    
    for i in tqdm(range(min_count)):
        source = str(real_files[i])
        file_name = source.split("/")[-1].split(".")[0]
        os.makedirs(f"{data_path}/face/resize/256/real/", exist_ok=True)
        dest = f"{data_path}/face/resize/256/real/{i}.png"
        resize_keep_all_content(source,dest)

if __name__ == "__main__":
    main()
