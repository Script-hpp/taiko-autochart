import os
import shutil

def is_leaf_directory(path):
    """Returns True if the directory has no subdirectories."""
    return all(not os.path.isdir(os.path.join(path, entry)) for entry in os.listdir(path))

def copy_from_leaf_folders_preserve_subdir(root_folder, destination_folder):
    """
    Copies files from all leaf folders under root_folder into
    destination_folder/<leaf_folder_name>/ preserving one subfolder.
    """
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for current_root, dirs, _ in os.walk(root_folder):
        if is_leaf_directory(current_root):
            leaf_folder_name = os.path.basename(current_root)
            dest_subdir = os.path.join(destination_folder, leaf_folder_name)
            os.makedirs(dest_subdir, exist_ok=True)

            for file in os.listdir(current_root):
                file_path = os.path.join(current_root, file)
                if os.path.isfile(file_path):
                    dest_path = os.path.join(dest_subdir, file)

                    # Handle filename conflicts in that folder
                    if os.path.exists(dest_path):
                        base, ext = os.path.splitext(file)
                        counter = 1
                        while os.path.exists(dest_path):
                            new_name = f"{base}_{counter}{ext}"
                            dest_path = os.path.join(dest_subdir, new_name)
                            counter += 1

                    shutil.copy2(file_path, dest_path)
                    print(f"Copied: {file_path} -> {dest_path}")

# === SET YOUR PATHS HERE ===
root_folder = r'D:\taiko_ai\ESE'
destination_folder = r'D:\taiko_ai\taiko-autochart\dataset-dirty'

copy_from_leaf_folders_preserve_subdir(root_folder, destination_folder)
