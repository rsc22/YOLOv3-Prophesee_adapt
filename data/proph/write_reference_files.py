import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
imgs_path = os.path.join(dir_path, r'image\train')

with open(dir_path+r'\train.txt', 'w') as f:
    img_names = os.listdir(imgs_path)
    for img in img_names:
        #print(os.path.join(imgs_path, img))
        f.write(os.path.join(imgs_path, img) + '\n')
