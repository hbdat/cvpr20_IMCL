# Interactive Multi-Label CNN Learning with Partial Labels

## Overview
This repository contains the implementation of [Interactive Multi-Label CNN Learning with Partial Labels](http://khoury.neu.edu/home/eelhami/publications/InteractiveCMLL-CVPR20.pdf).
> In this work, we address efficient end-to-end learning a multi-label CNN classifier with partial labels using an interactive dependency learning scheme.

![Image](https://github.com/hbdat/cvpr20_IMCL/raw/master/fig/interactive_sim_classifer.png)

---
## Prerequisites
+ Python 3.x
+ Tensorflow 1.x.x
+ sklearn
+ matplotlib
+ skimage
+ scipy

---
## Data Preparation

### Open Images

1) Please download pretrained Open Images model(https://storage.googleapis.com/openimages/2017_07/oidv2-resnet_v1_101.ckpt.tar.gz) into './model/resnet' folder

2) Please download Open Images urls and annotation into `./data/OpenImages` folder according to the instructions within the folder `./data/OpenImages/2017_11`.

3) To crawl images from the web, please run the script:
```
python ./download_imgs/asyn_image_downloader.py 					#`data_set` == `train`: download images into `./image_data/OpenImages/train/`
python ./download_imgs/asyn_image_downloader.py 					#`data_set` == `validation`: download images into `./image_data/OpenImages/validation/`
python ./download_imgs/asyn_image_downloader.py 					#`data_set` == `test`: download images into `./image_data/OpenImages/test/`
```
Please change the `data_set` variable in the script to `train`, `validation`, and `test` to download different data splits.

4) To extract features into TensorFlow storage format, please run:
```
python ./extract_data/extract_feature_2_TFRecords_OpenImages.py						#`data_set` == `train`: create `./TFRecords/train_feature.tfrecords`
python ./extract_data/extract_feature_2_TFRecords_OpenImages.py						#`data_set` == `validation`: create `./TFRecords/validation_feature.tfrecords`
python ./extract_data/extract_feature_2_TFRecords_OpenImages.py			        		#`data_set` == `test`:  create `./TFRecords/test_feature.tfrecords`
```
Please change the `data_set` variable in the `extract_feature_2_TFRecords_OpenImages.py` script to `train`, and `validation` to extract features from different data splits.

5) Please download sparse dictionary (https://drive.google.com/file/d/1he4omIq8N6SEuysFMNFeMkUlS7xpFIYE/view?usp=sharing) into the folder `./TFRecord/`


### CUB

1) Please download the following data files (https://drive.google.com/file/d/1gBQ_PQ0U8kzCaiiF7CvG92f1Ssfk8Zgq/view?usp=sharing), (https://drive.google.com/file/d/1fiNtiBj3hCj75eLHN1-02yWSDZrJ7GeN/view?usp=sharing), (https://drive.google.com/file/d/1O-0HTTFE9QpdTSQ8fdg31PsPAzJEiJga/view?usp=sharing) into `./TFRecord/` folder

### MSCOCO

1) Please download MSCOCO images and annotation into `./image_data/MSCOCO` folder according to the instructions within the folders`./image_data/MSCOCO/train2014`,`./image_data/MSCOCO/val2014`,`./image_data/MSCOCO/annotation`, and `./data/MSCOCO_1k`.

2) To extract features into TensorFlow storage format, please run:
```
python ./extract_data/extract_train_img_2_TFRecords_MSCOCO.py						#create ./TFRecord/train_MSCOCO_img_ZLIB.tfrecords
python ./extract_data/extract_test_img_2_TFRecords_MSCOCO.py						#create ./TFRecord/test_MSCOCO_img_ZLIB.tfrecords
python ./extract_data/extract_dic_img_2_TFRecords_MSCOCO.py							#create ./TFRecord/dic_10_MSCOCO_img_ZLIB.tfrecords
```

---
## Training and Evaluation

### Open Images

1) To pretrain the logistic backbone network, please run the script:
```
python ./OpenImages_experiments/baseline_logistic_OpenImages.py					# fixed feature representation
python ./OpenImages_experiments/e2e_baseline_logistic_OpenImages.py				# end-to-end training
```

2) To train our method, please run the script:
```
python ./OpenImages_experiments/interactive_learning_OpenImages.py				# fixed feature representation
python ./OpenImages_experiments/e2e_interactive_learning_OpenImages.py			# end-to-end training
```

3) To evaluate the performance, please run the script:
```
python ./evaluation/evaluation_interactive_learning_OpenImages.py				# fixed feature representation
python ./evaluation/evaluation_e2e_interactive_learning_OpenImages.py			# end-to-end training
```

### CUB
1) Please download the ImageNet ResNet backbone (http://download.tensorflow.org/models/resnet_v2_101_2017_04_14.tar.gz) into `./model/resnet_CUB`

2) To pretrain the logistic backbone network, please run the script:
```
python ./CUB_experiments/e2e_baseline_logistic_CUB.py
```

3) To train and evaluate our method, please run the script:
```
python ./CUB_experiments/e2e_interactive_learning_CUB.py
```

### MSCOCO

1) Please download the ImageNet VGG backbone (http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz) into `./model/vgg_ImageNet`

2) To pretrain the logistic backbone network, please run the script:
```
python ./MSCOCO_experiments/e2e_baseline_logistic_MSCOCO_vgg.py
```

3) To train and evaluate our method, please run the script:
```
python ./MSCOCO_experiments/e2e_interactive_learning_MSCOCO.py
```

---
## Citation
If this code is helpful for your research, we would appreciate if you cite the work:
```
@article{Huynh-mll:CVPR20,
  author = {D.~Huynh and E.~Elhamifar},
  title = {Interactive Multi-Label {CNN} Learning with Partial Labels},
  journal = {{IEEE} Conference on Computer Vision and Pattern Recognition},
  year = {2020}}
```
