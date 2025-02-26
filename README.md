# Continual Panoptic Perception

The implementation of *Continual Panoptic Perception* model.

## Preparation

### Requirement
The virtual environment follows [mmdetection](https://github.com/open-mmlab/mmdetection) is OK.

### Dataset
Put FineGrip dataset in `root/data/` as follows.
 ```
FineGrip
├── Images
│   ├── Train_panoptic
│   ├── Val_panoptic
├── Annotations
│   ├── Train_panoptic
│   ├── Val_panoptic
│   ├── Train_panoptic.json
│   ├── Val_panoptic.json
│   ├── Train_caption.json
│   ├── Val_caption.json
│   ├── Train_seg.json
│   ├── Val_seg.json
 ```


## Models

### FineGrip
 | task   | Download|
 | :----: | :----: |
 | 20-5   |  [link]( ) |
 | 15-5   |  [link]( ) |
 | 15-2   |  [link]( ) |
 | 10-5   |  [link]( ) |


## Run
`bash CPP_train.sh configs/PP/CPP.py path/to/checkpoint`
