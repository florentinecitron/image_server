# sudo docker run -it --rm hawaku/azcopy azcopy copy https://penzhanwu2.blob.core.windows.net/sgg/sgg_benchmark/vinvl_model_zoo/vinvl_vg_x152c4.pth .

cd models

# vinvl model
wget --show-progress https://penzhanwu2.blob.core.windows.net/sgg/sgg_benchmark/vinvl_model_zoo/vinvl_vg_x152c4.pth

# vinvl labelmap file
wget --show-progress https://penzhanwu2.blob.core.windows.net/sgg/sgg_benchmark/vinvl_model_zoo/VG-SGG-dicts-vgoi6-clipped.json

# vqa model
wget --show-progress https://biglmdiag.blob.core.windows.net/vinvl/model_ckpts/vqa/large/best.zip

# captioning model
wget --show-progress https://biglmdiag.blob.core.windows.net/vinvl/model_ckpts/image_captioning/coco_captioning_large_scst.zip