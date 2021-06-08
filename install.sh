
git clone https://github.com/microsoft/scene_graph_benchmark
git clone --recursive https://github.com/microsoft/Oscar/

conda create --name sg_benchmark python=3.7 -y
conda activate sg_benchmark
conda install ipython h5py nltk joblib jupyter pandas scipy
pip install ninja yacs==0.1.8 cython matplotlib tqdm opencv-python numpy=1.19.5
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
conda install -c conda-forge pycocotools
python -m pip install cityscapesscripts
conda install -c conda-forge nvidia-apex
cd scene_graph_benchmark
python setup.py build develop
python -m pip install tqdm yacs
pip install --ignore-installed --global-option='--with-libyaml' pyyaml
conda deactivate
cd -

conda create --name oscar python=3.7
conda activate oscar
conda install -c anaconda openjdk
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
pip install -r requirements.txt
cd Oscar
python setup.py build develop
pip install -r requirements.txt
pip install flask pandas
cd coco_caption
./get_stanford_models.sh
conda deactivate
cd -

git clone https://github.com/clp-research/image-description-sequences