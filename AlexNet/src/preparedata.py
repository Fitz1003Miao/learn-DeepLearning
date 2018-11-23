import cv2
import argparse
import os

def main():
    parse = argparse.ArgumentParser()
    parse.add_argument("--folder", "-f", help = "Path to image data", required = True)
    parse.add_argument("--output", "-o", help = "Path to output image data", required = True)

    args = parse.parse_args()
    print(args)

    folder = args.folder

    for img in os.listdir(folder):
        origin_img_data = cv2.imread(os.path.join(folder, img), -1)
        origin_img_shape = origin_img_data.shape
        
        



if __name__ == "__main__":
    main()