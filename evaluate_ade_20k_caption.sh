
conda activate oscar
mkdir -p caption_eval caption_eval_sequence

python captioning/evaluate_ade20k_caption.py evaluate_captions \
    image-description-sequences/data/captions.csv \
    caption_output \
    ADE_image_list.txt \
    caption_eval

python captioning/evaluate_ade20k_caption.py evaluate_sequences \
    image-description-sequences/data/sequences.csv \
    caption_output \
    caption_eval_sequence