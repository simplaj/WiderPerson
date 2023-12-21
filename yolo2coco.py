import json
import os
import cv2

def yolo_to_coco(yolo_dir, img_dir, output_file):
    data = {}
    data['images'] = []
    data['type'] = 'instances'
    data['annotations'] = []
    data['categories'] = []
    
    class_id = 0
    annotation_id = 0 
    image_id = 0
    for root, _, files in os.walk(yolo_dir):
        for file in files:
            image_id = int(file.rsplit(".", 1)[0])
            image = {}
            image['file_name'] = file.rsplit(".", 1)[0] + '.jpg'
            img2 = cv2.imread(os.path.join(img_dir, image['file_name']))
            image['height'], image['width'], _ = img2.shape
            image['id'] = image_id
            data['images'].append(image)
            
            f = open(os.path.join(root, file), 'r')
            for line in f:
                cat, x, y, w, h = list(map(float, line.strip().split()))
                annotation = {}
                cat = int(cat)
                annotation["iscrowd"] = 0
                annotation["image_id"] = image_id
                annotation['bbox'] = [x*image['width']-0.5*w*image['width'],y*image['height']-0.5*h*image['height'],w*image['width'],h*image['height']]
                annotation['area'] = w*image['width']*h*image['height']
                annotation['category_id'] = cat
                annotation['id'] = annotation_id
                annotation_id += 1
                annotation['ignore'] = 0
                annotation['segmentation'] = []
                data['annotations'].append(annotation)
                
        for i in range(5):   # assuming you have 80 classes 
            label_map = ['pedestrians', 'riders', 'partially-visible persons', 'ignore regions', 'crowd']
            category = {}
            category['supercategory'] = 'none'
            category['id'] = i
            category['name'] = i  # label_map[i]
            data['categories'].append(category)
    json.dump(data, open(output_file,'w'))

# call the function
yolo_to_coco("/home/tzh/Project/WiderPerson/data/WiderPerson/labels/test", "/home/tzh/Project/WiderPerson/data/WiderPerson/images/test", "output.json")