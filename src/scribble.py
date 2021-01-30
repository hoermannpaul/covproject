from main import preprocess_data
import os
if __name__ == "__main__":
    print(os.getcwd())
    DATA_DIR = "data/projectB_data/images"
    SIZE = 256, 144
    preprocess_data(DATA_DIR, SIZE)
    #main()