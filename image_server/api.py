import base64
import json
import pickle
import os.path
import tempfile
import logging
import sys
import sqlite3

import numpy as np
import torch
import pandas as pd

from flask import Flask, request
from flask import g

from oscar.modeling.modeling_bert import ImageBertForSequenceClassification
from oscar.run_vqa import VQADataset
from oscar.utils.task_utils import (_truncate_seq_pair,
                                    convert_examples_to_features_vqa,
                                    output_modes, processors)
from torch.utils.data import (DataLoader, Dataset, RandomSampler,
                              SequentialSampler, TensorDataset)
from transformers.pytorch_transformers import (WEIGHTS_NAME, BertConfig,
                                               BertTokenizer)


def get_examples_labels(processor, label_file, examples_data_dir, example_file):
    examples = processor.get_test_examples(examples_data_dir, example_file)
    labels = processor.get_labels(label_file)
    return examples, labels


class SingleImageDataset(VQADataset):

    @classmethod
    def create(cls, image_feature, image_id, question, processor, tokenizer, label_file, args):
        # image_id ADE_train_00002600.jpg

        # write question to tmpfile
        image_prefix = image_id.split("/", 1)[0]

        tf = tempfile.NamedTemporaryFile("w", delete=False)
        q_data = [{"q": question, "o": "", "img_key": image_id, "img_id": image_id, "q_id":"0", "label":[]}]
        print(json.dumps(q_data), file=tf)
        tf.close()
        examples_data_dir, example_file = os.path.split(tf.name)
        img_features = {
            str(image_id): torch.tensor(np.frombuffer(base64.b64decode(image_feature), dtype=np.float32).reshape((-1, 2054)))
        }

        return cls(tokenizer, img_features, processor, label_file, examples_data_dir, example_file, args)

    def __init__(self, tokenizer, img_features, processor, label_file, examples_data_dir, example_file, args):
        self.args = args

        self.tokenizer = tokenizer

        self.img_features = img_features

        self.examples, self.labels = get_examples_labels(
            processor,
            label_file=label_file,
            examples_data_dir=examples_data_dir,
            example_file=example_file
        )
        label_map = {label: i for i, label in enumerate(self.labels)}

        self.label_map = label_map
        logging.info(self.img_features.keys())
        self.features = self.tensorize(
            cls_token_at_end=False,
            cls_token=self.tokenizer.cls_token,
            sep_token=self.tokenizer.sep_token,
            cls_token_segment_id=0,
            pad_on_left=False,
            pad_token_segment_id=0
        )

    def get_img_feature(self, image_id):
        """ decode the image feature """
        logging.info(self.img_features.keys())
        return self.img_features[image_id]


class args:
    batch_size=10
    do_lower_case = True
    img_feature_dim = 2054
    img_feature_fomat = "pt"
    img_feature_type = "faster_r-cnn"

    image_folder = "oscar_input"

    label_file = "labels/trainval_ans2label.pkl"
    label2ans_file = 'labels/trainval_label2ans.pkl'

    #label_file = "/home/nrg/projects/vqa-test/vqa-data/trainval_ans2label.pkl"
    #label2ans_file = '/home/nrg/projects/vqa-test/vqa-data/trainval_label2ans.pkl'
    load_fast = True
    max_img_seq_length = 50
    max_seq_length = 128
    model_name_or_path = "/home/users/ngow/project/image_server/models/best_vqa"
    model_type = "bert"
    output_mode = "classification"
    task_name = "vqa_text"
    use_vg_dev = False


def create_app(db_path):

    def get_db():
        db = getattr(g, '_database', None)
        if db is None:
            db = g._database = sqlite3.connect(db_path)
        return db

    processor = processors[args.task_name]()
    label_list = processor.get_labels(args.label_file)
    num_labels = len(label_list)

    config_class, model_class, tokenizer_class = BertConfig, ImageBertForSequenceClassification, BertTokenizer
    config = config_class.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels, finetuning_task=args.task_name,
    )
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(args.model_name_or_path)
    model.eval()

    with open(args.label2ans_file, 'rb') as i:
        label2ans = pickle.load(i)

    app = Flask(__name__)

    @app.route('/answer_question')
    def answer_question():
        # curl 'localhost:5000/answer_question?image=ADE_train_00001505&question=what+color+is+the+building?'

        image_id = request.args.get('image')
        question = request.args.get('question')

        cur = get_db().cursor()
        cur.execute("select image_feature from image_features where image_id = :image_id limit 1", {"image_id": image_id})
        image_feature = bytes(cur.fetchone()[0], "ascii")

        test_dataset = SingleImageDataset.create(image_feature, image_id, question, processor, tokenizer, args.label_file, args)
        eval_sampler = SequentialSampler(test_dataset)
        eval_dataloader = DataLoader(test_dataset, sampler=eval_sampler, batch_size=args.batch_size)

        for batch in eval_dataloader:
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'token_type_ids': batch[2],
                'labels': None,
                'img_feats': batch[3]
            }

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs[0]
                val, idx = logits.max(1)

                for i in range(idx.size(0)):
                    answer = label2ans[test_dataset.labels[idx[i].item()]]
                    return answer

    @app.route('/get_caption')
    def get_caption():
        # curl 'localhost:5000/get_caption?image=ADE_train_00001505'
        image_id = request.args.get('image')
        cur = get_db().cursor()
        cur.execute("select caption from captions where image_id = :image_id limit 1", {"image_id": image_id})
        caption = cur.fetchone()[0]
        return caption

    return app



if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    db_path = sys.argv[1]
    create_app(db_path).run()