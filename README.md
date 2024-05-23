# Instrument anticipation

This repository contains code for the paper [Robotic scrub nurse as mind reader: Anticipating required surgical instruments based on real-time laparoscopic video analysis](https://www.nature.com/commsmed).
***
# Data structure

The directory containing the dataset should be organized like this:

```
-- prepro
    |-- anno
        |-- video_1.csv
        |-- video_2.csv
        |-- ...
    |-- data_split
        |-- all.txt
        |-- test.txt
        |-- train.txt
        |-- val.txt
    |-- image_tensors
        |-- video_1
            |-- video1_f0.pt
            |-- video1_f25.pt
            |--...
    |-- yolo_detect
        |-- video_1
            |-- labels
                |-- video1_f0.txt
                |-- video1_f25.txt
                |--...
    |-- yolo_segment
            |-- video_1
            |-- results
                |-- video1_f0_cls.pt
                |-- video1_f0_mask.pt
                |--...
```
As stated in the [paper](https://www.nature.com/commsmed) the data sets of the study are not publicly available due to restrictions related to privacy concerns for the research participants but are available from the corresponding author on reasonable request.
***
# Setup environment

```
pip install -r requirements.txt
```
***
# Training the models

Run the following commands to train the model.

## Stage 1 - Train Visual Feature Extractor

```
python train.py -c modules/cnn/config/visual_feature_module.yml
```

## Stage 2 - Train Temporal Multi-Frame Model

Run **ltc**, **mstcn** by replacing {}.
```
python train.py -c modules/temporal/config/{}.yaml
```

Edit the yaml-files for adjusting the hyperparamters.

***
