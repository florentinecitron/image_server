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

import pandas as pd
import itertools as it
import os.path


def main(annotations_file, image_feature_dir, out_dir):
    # annotations_file = "/home/nrg/projects/vqa-test/image_server/image-description-sequences/data/sequences.csv"

    df_ann = pd.read_csv(
        annotations_file,
        names="ignore seq_id image_id image_path image_cat image_subcat d1 d2 d3 d4 d5".split(),
        delimiter="\t",
        skiprows=1
    ).drop("ignore", axis=1).sort_values("image_path")

    im_features, labels = [], []

    for g, ims in it.groupby(df_ann.itertuples(), key=lambda x: x.image_path.split("/")[:2]):

        # open feature_file, label_file
        d = os.path.join(image_feature_dir, g)


        # find all ims in files and append to lists

        print(g)

    pass