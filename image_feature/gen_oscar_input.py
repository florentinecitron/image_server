import sys
import pandas as pd
import ast
import json
import base64
import numpy as np
import yaml
import os.path as op
import os
from maskrcnn_benchmark.structures.tsv_file_ops import tsv_reader, tsv_writer
import yaml
import os.path as op


def main():
    gen_oscar_input(sys.argv[1], sys.argv[2], sys.argv[3])


def gen_oscar_input(HW_FILE, SG_PREDICTIONS_FILE, OUTPUT_DIR):
    # https://gist.github.com/EddieKro/903ad08e85d670ff2b140a888d8c67c0

    #HW_FILE = '/home/nrg/projects/vqa-test/scene_graph_benchmark/tools/mini_tsv/data/train.hw.tsv'
    #SG_PREDICTIONS_FILE = "/home/nrg/projects/vqa-test/scene_graph_benchmark/tools/mini_tsv/data/output/inference/train/predictions.tsv"
    #OUTPUT_DIR = '/home/nrg/projects/vqa-test/scene_graph_benchmark/tools/mini_tsv/data/output/inference_test/'


    np.set_printoptions(suppress=True, precision=4)

    LABEL_FILE = op.join(OUTPUT_DIR,'label.tsv')
    FEATURE_FILE = op.join(OUTPUT_DIR,'feature.tsv')
    OUTPUT_YAML_FILE = op.join(OUTPUT_DIR, 'vinvl_test_yaml.yaml')
    if not op.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"path to {OUTPUT_DIR} created")

    hw_df = pd.read_csv(HW_FILE,sep='\t',header=None,converters={1:ast.literal_eval},index_col=0)
    df = pd.read_csv(SG_PREDICTIONS_FILE,sep='\t',header = None,converters={1:json.loads}) #converters={1:ast.literal_eval})
    df[1] = df[1].apply(lambda x: x['objects'])

    def generate_additional_features(rect,h,w):
        mask = np.array([w,h,w,h],dtype=np.float32)
        rect = np.clip(rect/mask,0,1)
        res = np.hstack((rect,[rect[3]-rect[1], rect[2]-rect[0]]))
        return res.astype(np.float32)

    def generate_features(x):
        idx, data,num_boxes = x[0],x[1],len(x[1])
        h,w,features_arr = hw_df.loc[idx,1][0]['height'],hw_df.loc[idx,1][0]['width'],[]

        for i in range(num_boxes):
            features = np.frombuffer(base64.b64decode(data[i]['feature']),np.float32)
            pos_feat = generate_additional_features(data[i]['rect'],h,w)
            x = np.hstack((features,pos_feat))
            features_arr.append(x.astype(np.float32))

        features = np.vstack(tuple(features_arr))
        features = base64.b64encode(features).decode("utf-8")
        return {"features":features, "num_boxes":num_boxes}

    def generate_labels(x):
        data = x[1]
        res = [{"class":el['class'].capitalize(),"conf":el['conf'], "rect": el['rect']} for el in data]
        return res

    df['feature'] = df.apply(generate_features,axis=1)
    df['feature'] = df['feature'].apply(json.dumps)

    df['label'] = df.apply(generate_labels,axis=1)
    df['label'] = df['label'].apply(json.dumps)

    tsv_writer(df[[0,'label']].values.tolist(),LABEL_FILE)
    tsv_writer(df[[0,'feature']].values.tolist(),FEATURE_FILE)

    yaml_dict = {
        "label": "label.tsv",
        "feature": "feature.tsv"
    }

    with open(OUTPUT_YAML_FILE, 'w') as file:
        yaml.dump(yaml_dict, file)

if __name__ == '__main__':
    main()