import shutil
import os
import sys
from pathlib import Path


def copy_common_files(folder_a: str, folder_b: str, folder_c: str) -> None:
    path_a = Path(folder_a)
    path_b = Path(folder_b)
    path_c = Path(folder_c)

    if not path_a.is_dir():
        sys.exit(f"Error: folder A '{folder_a}' does not exist or is not a directory.")
    if not path_b.is_dir():
        sys.exit(f"Error: folder B '{folder_b}' does not exist or is not a directory.")

    path_c.mkdir(parents=True, exist_ok=True)

    names_in_b = {f.name for f in path_b.iterdir() if f.is_file()}

    copied = 0
    for file in path_a.iterdir():
        if file.is_file() and file.name in names_in_b:
            shutil.copy2(file, path_c / file.name)
            print(f"Copied: {file.name}")
            copied += 1

    print(f"\nDone. {copied} file(s) copied to '{folder_c}'.")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python copy_common_files.py <folder_A> <folder_B> <folder_C>")
        sys.exit(1)

    copy_common_files(sys.argv[1], sys.argv[2], sys.argv[3])