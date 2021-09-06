# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license.

import sys

import glob
import os
import os.path as op
import json
import cv2
import base64
import tqdm
import yaml

from maskrcnn_benchmark.structures.tsv_file_ops import tsv_reader, tsv_writer
from maskrcnn_benchmark.structures.tsv_file_ops import generate_linelist_file
from maskrcnn_benchmark.structures.tsv_file_ops import generate_hw_file
from maskrcnn_benchmark.structures.tsv_file import TSVFile
from maskrcnn_benchmark.data.datasets.utils.image_ops import img_from_base64


which = sys.argv[1]

letter = sys.argv[2]

if len(letter) == 0:
    raise AssertionError("must include letter as argument [a-z] or special folder (misc, outliers)")

output_dir = f"image_tsv_files/{which}/{letter}_extra"

if not op.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

# To generate a tsv file:
data_path = f"/home/users/ngow/data/ImageCorpora/ADE20K_2016_07_26/images/{which}/{letter}/*/*/*.jpg"
img_list = glob.glob(data_path)
tsv_file = f"{output_dir}/val.tsv"
label_file = f"{output_dir}/val.label.tsv"
hw_file = f"{output_dir}/val.hw.tsv"
linelist_file = f"{output_dir}/val.linelist.tsv"
yaml_file = f"{output_dir}/val.yaml"

rows = []
rows_label = []
rows_hw = []
for img_p in tqdm.tqdm(img_list):
    img_key = img_p.split('.')[0]
    img_path = op.join(data_path, img_p)
    img = cv2.imread(img_path)
    img_encoded_str = base64.b64encode(cv2.imencode('.jpg', img)[1])

    # Here is just a toy example of labels.
    # The real labels can be generated from the annotation files
    # given by each dataset. The label is a list of dictionary
    # where each box with at least "rect" (xyxy mode) and "class"
    # fields. It can have any other fields given by the dataset.

    labels = []
    # TODO nick: are these used??
    # i don't think so because vinvl generates object labels and bounding boxes
    # but if you don't include this, test_sg_net fails.
    labels.append({"rect": [1, 1, 30, 40], "class": "ignore"})

    row = [img_key, img_encoded_str]
    rows.append(row)

    row_label = [img_key, json.dumps(labels)]
    rows_label.append(row_label)

    height = img.shape[0]
    width = img.shape[1]
    row_hw = [img_key, json.dumps([{"height":height, "width":width}])]
    rows_hw.append(row_hw)

tsv_writer(rows, tsv_file)
tsv_writer(rows_label, label_file)
tsv_writer(rows_hw, hw_file)

generate_linelist_file(label_file, save_file=linelist_file)

with open(yaml_file, "w") as o:
    d = {
        "img": "val.tsv",
        "label": "val.label.tsv",
        "hw": "val.hw.tsv",
        "labelmap": "val.labelmap.tsv",
        "linelist": "val.linelist.tsv"
    }
    yaml.dump(d, o)