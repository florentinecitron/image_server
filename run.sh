#####
# 1 #
#####



# do for a-z
#python image_feature/gen_ade_20k_image_tsv.py a
conda activate sg_benchmark
export WHICH=validation #training  # validation
ls ~/data/ImageCorpora/ADE20K_2016_07_26/images/${WHICH}/ | grep -v eaD | xargs -n 1 -I{} python image_feature/gen_ade_20k_image_tsv.py ${WHICH} {}

#####
# 2 #
#####

conda activate sg_benchmark
cd scene_graph_benchmark

# do for a-z
export WHICH=validation #training
export CUDA_AVAILABLE_DEVICES=1
mkdir ../image_feature_output_dir/${WHICH}
ls ~/data/ImageCorpora/ADE20K_2016_07_26/images/${WHICH}/ | \
    grep -v eaD | \
    xargs -n 1 -I{} \
    python tools/test_sg_net.py --config-file sgg_configs/vgattr/vinvl_x152c4.yaml \
    DATASETS.LABELMAP_FILE "../models/VG-SGG-dicts-vgoi6-clipped.json" \
    MODEL.WEIGHT "../models/vinvl_vg_x152c4.pth" \
    MODEL.ATTRIBUTE_ON False \
    TEST.IMS_PER_BATCH 1 \
    MODEL.ROI_HEADS.NMS_FILTER 1 \
    MODEL.ROI_HEADS.SCORE_THRESH 0.2 \
    DATA_DIR "../image_tsv_files/${WHICH}/{}/" \
    TEST.IGNORE_BOX_REGRESSION True \
    MODEL.ATTRIBUTE_ON True \
    OUTPUT_DIR "../image_feature_output_dir/${WHICH}/{}/" \
    DATASETS.TEST "('val.yaml',)" \
    DATASETS.FACTORY_TEST '("ODTSVDataset",)' \
    TEST.OUTPUT_FEATURE True \
    MODEL.DEVICE cuda \
    TEST.BBOX_AUG.ENABLED False


#####
# 3 #
#####

conda activate sg_benchmark

# do for a-z
export WHICH=training
ls ~/data/ImageCorpora/ADE20K_2016_07_26/images/${WHICH}/ | grep -v eaD | xargs -n 1 -I{} \
    python image_feature/gen_oscar_input.py \
    image_tsv_files/${WHICH}/{}/val.hw.tsv \
    image_feature_output_dir/${WHICH}/{}/inference/val.yaml/predictions.tsv \
    oscar_input/${WHICH}/{}/


#####
# 4 #
#####

conda activate oscar
cd Oscar

# batch size more than 8 causes shm (docker shared memory?) issues
export WHICH=training
ls ~/data/ImageCorpora/ADE20K_2016_07_26/images/${WHICH}/ | grep -v eaD | xargs -n 1 -I{} \
    python oscar/run_captioning.py \
    --do_test  \
    --test_yaml ../oscar_input/${WHICH}/{}/vinvl_test_yaml.yaml \
    --per_gpu_eval_batch_size 8 \
    --num_beams 5 \
    --max_gen_length 20 \
    --eval_model_dir ../models/coco_captioning_large_scst/checkpoint-4-50000 \
    --output_dir ../caption_output/${WHICH}/{}


#####
# 5
# generate sqlite file containing generated captions and image features for api
#####

conda activate oscar
python image_server/convert_sqlite.py \
    api.db \
    caption_output/ \
    oscar_input/


#####
# 6
# run api
#####

# run api

# example request
# $ curl 'localhost:5000/answer_question?image=a/aqueduct/ADE_train_00001505.jpg&question=what+do+you+see'
conda activate oscar
python image_server/api.py