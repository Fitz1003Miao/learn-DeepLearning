import cv2
import os
import h5py
import numpy as np

CLASS_NAME = ['cat', 'dog']

def argumentImage(img):
    argument_img_datas = []
    # for i in range(32):
    #     for j in range(32):
    argument_img_datas.append(img[0:224, 0:224])
    argument_img_datas.append(cv2.flip(img[0:224, 0:224], 1))
    return argument_img_datas

def resizeImage(origin_img, height, width):
    img_height, img_width = origin_img.shape[0:2]
    
    scale = min(img_height, img_width) / height
    new_img_height, new_img_width = int(img_height / scale), int(img_width / scale)
    new_img_data = cv2.resize(origin_img, (new_img_width, new_img_height), interpolation = cv2.INTER_AREA)

    return new_img_data

def saveDataFromList(img_list, outpath, flag = None, withType = True):
    img_data_list = []
    h5file_list = []
    
    if withType:
        img_type_list = []

    if not os.path.exists(outpath):
        os.makedirs(outpath, exist_ok = True)

    idx = 1
    for k, img in enumerate(img_list):
        img_data = cv2.imread(img, -1)
        if withType:
            img_type = CLASS_NAME.index(img.split('/')[-1].split('.')[0])
        
        argument_img_datas = argumentImage(img_data)
        print(img)
        for img in argument_img_datas:
            img_data_list.append(img)
            if withType:
                img_type_list.append(img_type)

        if (k + 1) % 1000 == 0 or k == len(img_list) - 1:

            h5filename = os.path.join(outpath, "%s_%d.h5" % (flag, idx))
            f = h5py.File(h5filename, 'w')
            f['data'] = img_data_list
            if withType:
                f['type'] = img_type_list
            f['data_num'] = len(img_data_list)
            f.close()

            idx += 1
            img_data_list.clear()
            if withType:
                img_type_list.clear()

            h5file_list.append(h5filename)
    saveList(h5file_list, os.path.join(outpath, "%s_file.txt" % (flag)))

def saveList(file_list, txt):
    with open(txt, 'w') as f:
        for file in file_list:
            f.write(file + "\n")
        f.close()

def load_data(file):
    with open(file, 'r') as f:
        points = []
        labels = []
        for line in f.readlines():
            path = line.strip('\n')
            data = h5py.File(path, 'r')
            points.append(data['data'][...].astype(np.float32))
            labels.append(data['type'][...].astype(np.int64))

        return (np.concatenate(points, axis = 0), np.concatenate(labels, axis = 0))

def shuffle(data, label):
    assert(data.shape[0] == label.shape[0])

    num = data.shape[0]
    indices = np.arange(0, num)
    np.random.shuffle(indices)

    return data[indices][...], label[indices][...]

    