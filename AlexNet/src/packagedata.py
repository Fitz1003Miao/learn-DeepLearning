import cv2
import argparse
import os
import random
import pickle
import shutil
import utils

def main():
    parse = argparse.ArgumentParser()
    parse.add_argument("--folder", "-f", help = "Path to image data", required = True)
    parse.add_argument("--outpath", "-o", help = "Path to output image data", required = True)

    args = parse.parse_args()
    print(args)

    folder = args.folder
    outpath = args.outpath

    train_img_list = [line.strip('\n') for line in open(os.path.join(folder, "train_file.txt"), 'r').readlines()]
    val_img_list = [line.strip('\n') for line in open(os.path.join(folder, "val_file.txt"), 'r').readlines()]
    test_img_list = [line.strip('\n') for line in open(os.path.join(folder, "test_file.txt"), 'r').readlines()]

    utils.saveDataFromList(train_img_list, outpath = os.path.join(outpath, 'train'), flag = "train", withType = True)
    utils.saveDataFromList(val_img_list, outpath = os.path.join(outpath, 'val'), flag = "val", withType = True)
    utils.saveDataFromList(test_img_list, outpath = os.path.join(outpath, 'test'), flag = "test", withType = False)

if __name__ == "__main__":
    main()