import cv2
import argparse
import os
import random
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

    types = ['train', 'test']
    os.makedirs(os.path.join(outpath, 'train'), exist_ok = True)
    os.makedirs(os.path.join(outpath, 'val'), exist_ok = True)
    os.makedirs(os.path.join(outpath, 'test'), exist_ok = True)

    # resize and crop image
    for type in types:
        for img in os.listdir(os.path.join(folder, type)):
            origin_img_data = cv2.imread(os.path.join(folder, type, img), -1)
            new_img_data = utils.resizeImage(origin_img_data, 256, 256)
            new_img_height, new_img_width = new_img_data.shape[0:2]

            new_img_data = new_img_data[(new_img_height // 2 - 128):(new_img_height // 2 + 128), (new_img_width // 2 - 128) : (new_img_width // 2 + 128)]
            cv2.imwrite(os.path.join(outpath, type, img), new_img_data)
        print(type, " has finished")

    # split train / val image
    train_img_list = [os.path.join(outpath, 'train', img) for img in os.listdir(os.path.join(outpath, 'train'))]
    random.shuffle(train_img_list)

    train_img_list, val_img_list = train_img_list[:int(len(train_img_list) * 0.8)], train_img_list[int(len(train_img_list) * 0.8):]
    for val_img in val_img_list:
        shutil.move(val_img, os.path.join(outpath, 'val', val_img.split('/')[-1]))

    train_img_list = [os.path.join(outpath, 'train', img) for img in os.listdir(os.path.join(outpath, 'train'))]
    val_img_list = [os.path.join(outpath, 'val', img) for img in os.listdir(os.path.join(outpath, 'val'))]
    test_img_list = [os.path.join(outpath, 'test', img) for img in os.listdir(os.path.join(outpath, 'test'))]
    
    utils.saveList(train_img_list, os.path.join(outpath, 'train_file.txt'))
    utils.saveList(val_img_list, os.path.join(outpath, 'val_file.txt'))
    utils.saveList(test_img_list, os.path.join(outpath, 'test_file.txt'))

if __name__ == "__main__":
    main()