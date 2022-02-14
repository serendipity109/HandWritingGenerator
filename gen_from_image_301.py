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
import random

wd = os.getcwd()
semantic_source = wd + '/utils/char_source/301.txt'
char_img_list_path = wd + '/utils/chnumengds.json'
# images_dst_folder = wd + '/temp_dst_folder'
images_dst_folder = '/home/sightour/Desktop/training_data/keras-chinese-ocr/train/images/tmp'
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


def trim_img(src_img, word_size, threshold = True):
    gray_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    ret, thr = cv2.threshold(gray_img, 100, 255, cv2.THRESH_BINARY)
    X = []
    Y = []
    for y, x in np.argwhere(thr == 0):
        X.append(x)
        Y.append(y)
    try:
        dst_img = src_img[max(0, min(Y) - 10): min(max(Y) + 10, 300), max(0, min(X) - 10): min(max(X) + 10, 300), :]
    except:
        return []
    resized_img = cv2.resize(dst_img, (word_size, word_size), interpolation=cv2.INTER_AREA)
    if threshold:
        ret, thr = cv2.threshold(resized_img, 180, 255, cv2.THRESH_BINARY)
        thr = cv2.cvtColor(thr, cv2.COLOR_BGR2GRAY)
        thr = cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR)
        return thr
    else:
        return resized_img


def comma_period(src_img, word_size):
    gray_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    ret, thr = cv2.threshold(gray_img, 180, 255, cv2.THRESH_BINARY)
    resized_img = cv2.resize(thr, (int(word_size*0.6), int(word_size*0.6)), interpolation=cv2.INTER_AREA)
    padded_img = cv2.copyMakeBorder(resized_img, int(word_size*0.4), 0, int(word_size*0.2), int(word_size*0.2), cv2.BORDER_CONSTANT, value=(255, 255, 255))
    resized_img2 = cv2.resize(padded_img, (word_size, word_size), interpolation=cv2.INTER_AREA)
    img2 = cv2.cvtColor(resized_img2, cv2.COLOR_GRAY2BGR)
    return img2


def dilate(src_img):
    kernel = np.ones((5, 5), np.uint8)
    dilated_img = cv2.erode(src_img, kernel, iterations=1)
    return dilated_img


def erode(src_img):
    kernel = np.ones((1, 1), np.uint8)
    eroded_img = cv2.dilate(src_img, kernel, iterations=1)
    return eroded_img


def load_char_img_dict():
    global char_img_dict
    with open(char_img_list_path, encoding='utf-8') as f:
        char_img_dict = json.load(f)


def load_char_list():
    global char_list
    with open(semantic_source, 'r', encoding="utf-8") as f:
        char_list = f.readlines()
    char_list = [c.replace('\n', '') for c in char_list]


def add_bg(img):
    bg_lst = glob.glob('/home/sightour/Documents/sightour3/general-python-projects/Tools/text_renderer-master/example_data/bg/*')
    bg_pth = random.sample(bg_lst, 1)[0]
    # print(bg_pth)
    bg = cv2.imread(bg_pth)
    h = img.shape[0]
    w = img.shape[1]
    try:
        y = random.sample([i for i in range(bg.shape[0] - h - 1)], 1)[0]
        x = random.sample([i for i in range(bg.shape[1] - w - 1)], 1)[0]
        bg = bg[y:y + h, x:x + w]
    except:
        bg = cv2.resize(bg, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
    t1 = 255 - img
    t2 = 255 - bg
    t2 = t2 * 0.9
    t3 = t1 + t2
    t4 = np.int16(t3)
    t5 = t4 - 255
    t6 = abs(t5)
    return np.uint8(t6)


load_char_img_dict()
load_char_list()


class HandWritingGenerator:

    def __init__(self, img_list):
        self.img_list = img_list
        self.dst_width = 280
        self.dst_height = 32

    def img_synthesis(self, filename):
        synthesised_img = cv2.hconcat(self.img_list)
        border_h = 300 - synthesised_img.shape[0]
        top = np.random.randint(low=10, high=border_h, size=1)[0]
        bottom = border_h - top
        border_w = 2625 - synthesised_img.shape[1]
        left = np.random.randint(low=10, high=border_w, size=1)[0]
        right = border_w - left
        padded_img = cv2.copyMakeBorder(synthesised_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        # cv2.imwrite(f'/home/sightour/Desktop/training_data/keras-chinese-ocr/train/images/tmp/{filename}_1.jpg', synthesised_img)
        synthesised_img = add_bg(padded_img)
        pil_img = Image.fromarray(synthesised_img)
        pil_img = pil_img.resize((self.dst_width, self.dst_height), Image.BICUBIC)
        return pil_img


def generate_img(img, q=None):
    global lock, counter, char_list, char_img_dict
    num = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
    filename = str(img).zfill(7)
    # Make sure different process has different random seed
    np.random.seed()

    line = char_list[0]
    length = np.random.randint(10, min(len(line), 20))
    start = np.random.randint(0, (len(line) - length + 1))
    chars = line[start: min(start + length, len(line))]

    selected_char_list = []
    selected_img_list = []

    word_size = np.random.randint(low=200, high=250, size=1)[0]
    text_len = 0
    word_src_lst = []
    for i, c in enumerate(chars):
        img_path = None
        try:
            img_path_list = char_img_dict[c]
        except KeyError:
            # if no image exist, replace by '一'
            c = '/'
            img_path_list = char_img_dict[c]
        while img_path is None:
            oo = np.random.randint(0, len(img_path_list))
            temp_img_path = img_path_list[oo]
            # To deal with unicode character in path
            src_img = cv2.imdecode(np.fromfile(temp_img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if c in num:
                eroded_img = erode(src_img)
                trimmed_img = cv2.resize(eroded_img, (word_size, word_size), interpolation=cv2.INTER_AREA)
                # trimmed_img = trim_img(resized_img, word_size)
                img_path = temp_img_path
            else:  # english and symbols
                trimmed_img = trim_img(src_img, word_size)
                if c in [',', '.']:
                    # if not os.path.exists(f'{images_dst_folder}/{filename}/'):
                    #     os.makedirs(f'{images_dst_folder}/{filename}/')
                    # cv2.imwrite(f'{images_dst_folder}/{filename}/{i}_t1.jpg', trimmed_img)
                    trimmed_img = comma_period(trimmed_img, word_size)
                    # cv2.imwrite(f'{images_dst_folder}/{filename}/{i}_t2.jpg', trimmed_img)
                if len(trimmed_img) == 0:
                    break
                if np.sum(trimmed_img) == np.sum([np.ones(trimmed_img.shape, dtype=int)*255]):
                    cv2.imwrite('./src_img.jpg', src_img)
                    # cv2.imwrite('./dilated_img.jpg', dilated_img)
                    cv2.imwrite('./trimmed_img.jpg', trimmed_img)
                img_path = temp_img_path
        if len(trimmed_img) > 0:
            selected_char_list.append(c)
            text_len += trimmed_img.shape[1]
            word_src_lst.append('/'.join(img_path.split('/')[-2:]))
            selected_img_list.append(trimmed_img)
            if text_len > 2000:
                print(filename, word_src_lst)
                # if not os.path.exists(f'{images_dst_folder}/{filename}/'):
                #     os.makedirs(f'{images_dst_folder}/{filename}/')
                # for i, img in enumerate(selected_img_list):
                #     cv2.imwrite(f'{images_dst_folder}/{filename}/{i}.jpg', img)
                break


    hw = HandWritingGenerator(selected_img_list)
    ea = effect_utils.EffectApplier()
    ea.initialize_effect()
    text_img = hw.img_synthesis(filename)
    # text_img.save(images_dst_folder + "/301" + filename + '_2.jpg')
    dst_img = ea.do_variation(text_img, filename)
    dst_img.save(images_dst_folder + "/301" + filename + '.jpg')
    text = u''.join(selected_char_list)
    line = "301" + filename + '.jpg ' + text

    if q is not None:
        q.put(line)


if __name__ == "__main__":
    manager = mp.Manager()
    q = manager.Queue()

    # gen_images_num = 100
    gen_images_num = 300000

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
