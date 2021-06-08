import sqlite3
import glob
import os.path
import csv
import json
import sys
import itertools as it
csv.field_size_limit(sys.maxsize)


def main(db_path, caption_path, image_feature_path):

    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("create table captions (image_id, caption)")
    cur.execute("create table image_features (image_id, image_feature)")

    def get_image_features():
        glob_path = os.path.join(image_feature_path, "*/*/feature.tsv")
        print(glob_path)
        for f in glob.glob(glob_path):
            with open(f) as i:
                reader = csv.reader(i, delimiter="\t")
                for full_name, json_str in reader:
                    name = full_name.rsplit("/",1)[1]
                    yield name, json.loads(json_str)["features"]

    def get_captions():
        glob_path = os.path.join(caption_path, "*/*/pred.coco_caption.vinvl_test_yaml.beam5.max20.odlabels.tsv")
        print(glob_path)
        for f in glob.glob(glob_path):
            with open(f) as i:
                reader = csv.reader(i, delimiter="\t")
                for full_name, json_str in reader:
                    name = full_name.rsplit("/",1)[1]
                    yield name, json.loads(json_str)[0]["caption"]

    for _, els in it.groupby(enumerate(get_image_features()), lambda idx_els: idx_els[0]//200):
        data = [e[1] for e in els]
        print(f"inserted {len(data)} elements into image_features")
        cur.executemany("insert into image_features values (?, ?)", data)

    for _, els in it.groupby(enumerate(get_captions()), lambda idx_els: idx_els[0]//200):
        data = [e[1] for e in els]
        print(f"inserted {len(data)} elements into captions")
        cur.executemany("insert into captions values (?, ?)", data)

    con.commit()
    con.close()

if __name__ == "__main__":
    db_path = sys.argv[1]
    caption_path = sys.argv[2]
    image_feature_path = sys.argv[3]
    main(db_path, caption_path, image_feature_path)