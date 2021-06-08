
conda activate oscar
mkdir -p caption_eval
python captioning/evaluate_ade20k_caption.py evaluate \
    image-description-sequences/data/sequences.csv \
    caption_output \
    caption_eval