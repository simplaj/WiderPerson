import brambox as bb
import numpy as np
import pandas as pd


pred = bb.io.load('det_coco', '/home/tzh/Project/WiderPerson/runs/detect/val3/predictions.json')
gt = bb.io.load('det_coco', '/home/tzh/Project/WiderPerson/output.json')
print(pred.head())
print(gt.head())