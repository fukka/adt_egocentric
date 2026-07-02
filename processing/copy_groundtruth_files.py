import shutil
import os
from pathlib import Path

root_1 = Path("/group-volume/Fengjia/data/projectaria_tools_adt_data")
root_2 = Path("/group-volume/Fengjia/data/projectaria_tools_adt_data_clean")

for src_gt in root_1.glob("*/groundtruth"):
    folder_name = src_gt.parent.name
    dst_gt = root_2 / folder_name / "groundtruth"
    if 'Apartment_release_clean_seq' in folder_name:
        dst_gt.mkdir(parents=True, exist_ok=True)
        for file in src_gt.iterdir():
            if file.is_file():
                shutil.copy2(file, dst_gt / file.name)
        print(f"Copied {src_gt} -> {dst_gt}")