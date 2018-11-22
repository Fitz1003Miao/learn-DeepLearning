#-*- coding: utf-8 -*-
import requests
import os
from tqdm import tqdm
import tarfile

def download_from_url(url, filepath):
    if os.path.exists(filepath):
        return False
    
    response = requests.get(url, stream = True)
    total_size = int(response.headers.get('content-length', 0))
    chunk_size = 1024 * 1024

    bars = total_size // chunk_size

    with open(filepath, 'wb') as f:
        for data in tqdm(response.iter_content(chunk_size = chunk_size), total = bars, desc = url.split('/')[-1], unit = 'M'):
            f.write(data)
    
    return True

def main():
    # download image_url
    image_url_url = "http://www.image-net.org/imagenet_data/urls/imagenet_spring10_urls.tgz"
    data_folder = "../data"
    
    if not os.path.exists(data_folder):
        os.makedirs(data_folder, exist_ok = True)

    image_url_tar_file = os.path.join(data_folder, image_url_url.split("/")[-1])
    isdownload = download_from_url(image_url_url, image_url_tar_file)
    
    # extract image_url
    if isdownload:
        tar = tarfile.open(image_url_tar_file)
        tar.extractall(path = data_folder)
        os.remove(image_url_tar_file)

    image_url_txt_file = os.path.join(data_folder, "spring10_urls.txt")

    img_folder = os.path.join(data_folder, 'img')
    if not os.path.exists(img_folder):
        os.makedirs(img_folder, exist_ok = True)
        img_types = set()

        with open(image_url_txt_file, "r") as f:
            for line in f.readlines():
                image_name, image_url = line.strip('\n').split('\t')
                img_type = image_url.split['.'][-1]

                isdownload = download_from_url(image_url, os.path.join(img_folder, image_name + "." + img_type))

                if isdownload:
                    img_types.add(img_type)
                    print(image_name, " has done")
                else:
                    print(image_name, " has passed")

            print(img_types)            
if __name__ == "__main__":
    main()