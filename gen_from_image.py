import traceback
import os
import glob
import time
from itertools import repeat
import numpy as np
import multiprocessing as mp
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import effect_utils
import json

wd = os.getcwd()
semantic_source = wd + '/utils/char_source/DRCD_sorted.txt'
char_img_list_path = wd + '/utils/std_6964_image_list.json'
# images_dst_folder = wd + '/temp_dst_folder'
images_dst_folder = 'D:/training_data/keras-chinese-ocr/train/images/007'
tmp_text_path = wd + "/tmp_text.txt"


fixed_char_length = 10
max_processes = os.cpu_count()
print("max_processes = {}, will use half core".format(max_processes))
num_of_processes = max_processes // 2


char_list = []
char_img_dict = {}

lock = mp.Lock()
counter = mp.Value('i', 0)
STOP_TOKEN = 'kill'


def start_listen(q, fname):
    """ listens for messages on the q, writes to file. """
    print(fname)
    f = open(fname, mode='a', encoding='utf-8')
    while 1:
        m = q.get()
        if m == STOP_TOKEN:
            break
        try:
            f.write(str(m) + '\n')
        except:
            traceback.print_exc()

        with lock:
            if counter.value % 1000 == 0:
                f.flush()
    f.close()


def sort_labels(tmp_label_fname, label_fname):
    with open(tmp_label_fname, mode='r', encoding='utf-8') as f:
        lines = f.readlines()

    lines = sorted(lines)
    with open(label_fname, mode='w', encoding='utf-8') as f:
        for line in lines:
            f.write(line)


def trim_img(src_img):
    gray_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    ret, thr = cv2.threshold(gray_img, 254, 255, cv2.THRESH_BINARY)
    thr = (255 - thr)
    rows, cols = np.nonzero(thr)
    try:
        left_most = cols.min()
        righ_most = cols.max()
    except ValueError:
        return src_img
    left_offset = np.random.randint(10, 50)
    righ_offset = np.random.randint(10, 50)
    dst_img = src_img[:, left_most - left_offset:righ_most + righ_offset, :]
    return dst_img


def load_char_img_dict():
    global char_img_dict
    with open(char_img_list_path, encoding='utf-8') as f:
        char_img_dict = json.load(f)


def load_char_list():
    global char_list
    with open(semantic_source, 'r', encoding="utf-8") as f:
        char_list = f.readlines()


load_char_img_dict()
load_char_list()


class HandWritingGenerator:

    def __init__(self, img_list):
        self.img_list = img_list
        self.dst_width = 280
        self.dst_height = 32

    def img_synthesis(self):
        synthesised_img = cv2.hconcat(self.img_list)
        pil_img = Image.fromarray(synthesised_img)
        pil_img = pil_img.resize((self.dst_width, self.dst_height), Image.BICUBIC)
        return pil_img


def generate_img(img_index, q=None):
    global lock, counter, char_list, char_img_dict
    filename = str(img_index).zfill(7)
    # Make sure different process has different random seed
    np.random.seed()

    start_idx = img_index * fixed_char_length
    try:
        chars = char_list[0][start_idx: start_idx + 10]
    except IndexError:
        return

    selected_char_list = []
    selected_img_list = []

    for c in chars:
        img_path = None
        try:
            img_path_list = char_img_dict[c]
        except KeyError:
            # if no image exist, replace by '一'
            c = '一'
            img_path_list = char_img_dict[c]
        while img_path is None:
            oo = np.random.randint(0, len(img_path_list))
            temp_img_path = img_path_list[oo]
            # To deal with unicode character in path
            src_img = cv2.imdecode(np.fromfile(temp_img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            trimmed_img = trim_img(src_img)
            # if the char image is too small
            if trimmed_img.shape[1] < 150:
                # print("current char: ", c)
                # print("image width: ", trimmed_img.shape[1])
                continue
            else:
                img_path = temp_img_path
        selected_char_list.append(c)
        selected_img_list.append(trimmed_img)

    hw = HandWritingGenerator(selected_img_list)
    ea = effect_utils.EffectApplier()
    ea.initialize_effect()
    text_img = hw.img_synthesis()
    dst_img = ea.do_variation(text_img)
    dst_img.save(images_dst_folder + "/007" + filename + '.jpg')
    text = u''.join(selected_char_list)
    line = "007" + filename + '.jpg ' + text

    if q is not None:
        q.put(line)


if __name__ == "__main__":
    manager = mp.Manager()
    q = manager.Queue()

    gen_images_num = len(char_list[0]) // fixed_char_length

    exist_img_list = glob.glob(images_dst_folder + "/*.jpg")
    exist_img_count = len(exist_img_list)
    start_index = exist_img_count
    end_index = start_index + gen_images_num

    start_time = time.time()
    with mp.Pool(processes=num_of_processes) as pool:
        pool.apply_async(start_listen, (q, tmp_text_path))
        pool.starmap(generate_img, zip(range(start_index, end_index), repeat(q)))
        q.put(STOP_TOKEN)
        pool.close()
        pool.join()
    end_time = time.time()
    print("Finish generate data, time elapsed = {}s".format(end_time - start_time))
    text_path = images_dst_folder + "/texts.txt"
    sort_labels(tmp_text_path, text_path)
