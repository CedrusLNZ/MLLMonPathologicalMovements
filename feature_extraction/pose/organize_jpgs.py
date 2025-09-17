import os
import shutil
import sys

def organize_images(directory):
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return
    
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            # parts = filename.split("_")
            parts = (filename[:-11],filename[-11:])
            if parts[1][:7].isdigit():  # Ensure correct format
                prefix = parts[0]
                subdir = os.path.join(directory, prefix)
                
                if not os.path.exists(subdir):
                    os.makedirs(subdir)
                
                src_path = os.path.join(directory, filename)
                dest_path = os.path.join(subdir, filename)
                
                shutil.move(src_path, dest_path)
                print(f"Moved {filename} -> {subdir}/")
    
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <directory_path>")
    else:
        organize_images(sys.argv[1])
