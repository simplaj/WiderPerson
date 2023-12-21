import brambox as bb
import numpy as np
import pandas as pd
import json


classes = ['pedestrians', 'riders', 'partially-visible persons', 'ignore regions', 'crowd']
def rm_other(json_path):
    with open(json_path, 'r') as f:
        datas = json.load(f)
    res = []
    for data in datas:
        if data['category_id'] == 0:
            data['category_id'] = classes[data['category_id']]
            res.append(data)
    with open('det.json', 'w') as f:
        f.writelines(json.dumps(res))

rm_other('/home/tzh/Project/WiderPerson/runs/detect/val4/predictions.json')
pred = bb.io.load('det_coco', 'det.json')
gt = bb.io.load('anno_coco', 'test.json')
mr_fppi = bb.stat.mr_fppi(pred, gt)
print(mr_fppi)
res = bb.stat.lamr(mr_fppi)
print(res)