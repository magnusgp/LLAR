import os


def cleanDataDir(dirpath, mode="trimmed"):
    """Cleans data directory to remove certain files from subdirectories.
       Defaults to removing trimmed files from the normal data directory.

    Args:
        dirpath (str): Path to the data directory.
        mode (str, optional): Cleaning mode, either "trimmed" or "raw". Defaults to "trimmed".
    """
    deleteCount = 0
    if mode == "trimmed":
        for root, dirs, files in os.walk(dirpath):
            for file in files:
                if file.endswith("_trimmed.npy"):
                    #os.remove(os.path.join(root, file))
                    print("Deleting file: ", os.path.join(root, file))
                    deleteCount += 1
    if mode == "trimmed_wav":
        for root, dirs, files in os.walk(dirpath):
            for file in files:
                if file.endswith("_trimmed.wav"):
                    #os.remove(os.path.join(root, file))
                    print("Deleting file: ", os.path.join(root, file))
                    deleteCount += 1
    elif mode == "raw":
        for root, dirs, files in os.walk(dirpath):
            for file in files:
                if file.endswith(".npy") and not file.endswith("_trimmed.npy"):
                    # careful with this!
                    #os.remove(os.path.join(root, file))
                    pass
    else:
        print("Invalid cleaning mode. Must be either 'trimmed' or 'raw'.")
        
    print(f"Deleted {deleteCount} files.")
        
if __name__ == "__main__":
    cleanDataDir("output/embeddings/ECS50", mode="trimmed")