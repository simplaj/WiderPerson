import os
from PIL import Image
import shutil
from tqdm import tqdm


# coding=utf-8
def check_charset(file_path):
    import chardet
    with open(file_path, "rb") as f:
        data = f.read(4)
        charset = chardet.detect(data)['encoding']
    return charset


def convert(size, box0, box1, box2, box3):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box0 + box2) / 2 * dw
    y = (box1 + box3) / 2 * dh
    w = (box2 - box0) * dw
    h = (box3 - box1) * dh
    return (x, y, w, h)


def process(mode):
    prefix = 'train' if mode == 'train' else 'val'
    
    outpath_txt = f'./data/WiderPerson/labels/{mode}'   
    outpath_jpg = f'./data/WiderPerson/images/{mode}'
    os.makedirs(outpath_txt, exist_ok=True)
    os.makedirs(outpath_jpg, exist_ok=True)

    path = f'./data/{prefix}.txt'
    with open(path, 'r') as f:
        img_ids = [x for x in f.read().splitlines()]
        
    begin = 500 if mode == 'test' else 0
    end = 500 if mode == 'val' else len(img_ids)

    for img_id in tqdm(img_ids[begin:end]):  # '000040'
        img_path = './data/Images/' + img_id + '.jpg'

        with Image.open(img_path) as Img:
            img_size = Img.size

        ans = ''

        label_path = img_path.replace('Images', 'Annotations') + '.txt'
        
        outpath = outpath_txt + "/" + img_id + '.txt'

        with open(label_path, encoding=check_charset(label_path)) as file:
            line = file.readline()
            count = int(line.split('\n')[0])  # 里面行人个数
            line = file.readline()
            while line:
                cls = int(line.split(' ')[0])
                # if cls == 1:
                if cls in [1, 3]:
                    cls_map = [0, 0, 0, 1, 1]
                    xmin = float(line.split(' ')[1])
                    ymin = float(line.split(' ')[2])
                    xmax = float(line.split(' ')[3])
                    ymax = float(line.split(' ')[4].split('\n')[0])
                    # print(img_size[0], img_size[1], xmin, ymin, xmax, ymax)
                    bb = convert(img_size, xmin, ymin, xmax, ymax)
                    ans = ans + f'{cls_map[cls]}' + ' ' + ' '.join(str(a) for a in bb) + '\n'
                line = file.readline()
        with open(outpath, 'w') as outfile:
            outfile.write(ans)
        # 想保留原文件用copy
        # shutil.copy(img_path, outpath_jpg + '/' + img_id + '.jpg')
        # 直接移动用这个
        # shutil.move(img_path, outpath_jpg + '/' + img_id + '.jpg')


if __name__ == '__main__':
    for mode in ['train', 'val', 'test']:
        process(mode)