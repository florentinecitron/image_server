"""
{'image_id': key,
  'caption': cap}

need:
    self.label_file = find_file_path_in_yaml(self.cfg['label'], self.root)
    self.feat_file = find_file_path_in_yaml(self.cfg['feature'], self.root)
    self.caption_file = find_file_path_in_yaml(self.cfg.get('caption'), self.root)

take the labeled data, then find all image ids, then create the reduced

also create a yaml file with those files listed
"""

import csv
import glob
import itertools as it
import json
import os.path

import pandas as pd

"""
    res_tsv: TSV file, each row is [image_key, json format list of captions].
        Each caption is a dict, with fields "caption", "conf".

    label_file: JSON file of ground truth captions in COCO format.
        {"annotations": [{"image_id": "", "id": "", ...?}]}
"""

from oscar.utils.caption_evaluate import evaluate_on_coco_caption
import fire


def evaluate(annotations_file, generated_caption_dir, out_dir):
    # annotations_file = "/home/nrg/projects/vqa-test/image_server/image-description-sequences/data/sequences.csv"
    # generated_caption_dir = "/project/image_server/caption_output"

    annotation_cols = "ignore seq_id image_id image_path image_cat image_subcat d1 d2 d3 d4 d5".split()
    df_ann = (
        pd.read_csv(
            annotations_file,
            names=annotation_cols,
            delimiter="\t",
            skiprows=1,
        )
        .drop("ignore", axis=1)
        .sort_values("image_path")
    )

    image_ids = set(df_ann.image_path)

    # create label.json
    #for k, _group in it.groupby(
    #    df_ann.itertuples(), key=lambda x: x.image_path.split("/")[:2]
    #):
    annotations = []
    for el in df_ann.itertuples():
        image_id = el.image_path
        caption_id = str(el.seq_id)
        caption = el.d1
        annotations.append(
            {"image_id": image_id, "id": caption_id, "caption": caption}
        )

    label_file = os.path.join(out_dir, "label.json")
    with open(label_file, "w") as o:
        print(json.dumps({"annotations": annotations, "images": [{"id": a["image_id"]} for a in annotations], "type": "", "info": "", "licenses": ""}), file=o)


    # create res_tsv
    tsv_out = []
    caption_file_glob = os.path.join(
        generated_caption_dir,
        "*/*/pred.coco_caption.vinvl_test_yaml.beam5.max20.odlabels.tsv",
    )
    for caption_file_path in glob.glob(caption_file_glob):
        with open(caption_file_path) as i:
            # /home/users/ngow/data/ImageCorpora/ADE20K_2016_07_26/images/training/a/access_road/ADE_train_00001022 \t [{"caption": "a road with a tree and a bench on the side of it.", "conf": 0.9170628786087036}]
            reader = csv.reader(i, delimiter="\t")
            for image_path, caption_json in reader:
                # image_id = "/home/users/ngow/data/ImageCorpora/ADE20K_2016_07_26/images/training/a/alley/ADE_train_00001274".rsplit("/", 4)
                image_id = "/".join(image_path.rsplit("/", 4)[1:]) + ".jpg"
                if image_id not in image_ids:
                    continue
                caption = json.loads(caption_json)[0]
                tsv_out.append((image_id, json.dumps([caption])))

    res_file = os.path.join(out_dir, "res.tsv")
    with open(res_file, "w") as o:
        for image_id, caption_json in tsv_out:
            print(f"{image_id}\t{caption_json}", file=o)

    evaluate_outfile = os.path.join(out_dir, "evaluate_result.json")
    evaluate_on_coco_caption(res_file, label_file, evaluate_outfile)


if __name__ == "__main__":
    fire.Fire()