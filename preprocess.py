import os
import glob
import cv2
import json


if __name__ == '__main__':
    src = '/home/sightour/Documents/sightour3/general-python-projects/Tools/handwritten_generator/emnist/A_Z'
    keymap = '/home/sightour/Documents/sightour3/general-python-projects/Tools/handwritten_generator/utils/chnum.json'
    with open(keymap, encoding='utf-8') as f:
        kmp = json.load(f)
    letters = os.listdir(src)
    for letter in letters:
        imgs = glob.glob(f'{src}/{letter}/*')
        tmp = []
        for img_pth in imgs:
            img = cv2.imread(img_pth)
            cvt_img = cv2.bitwise_not(img)
            cv2.imwrite(img_pth, cvt_img)
            tmp.append(img_pth)
        kmp.update({letter: tmp})

    with open('/home/sightour/Documents/sightour3/general-python-projects/Tools/handwritten_generator/utils/chnumeng.json', 'w', encoding='utf8') as fp:
        json.dump(kmp, fp, ensure_ascii=False, indent=4)
