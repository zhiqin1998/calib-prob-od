# Calibration of Probabilistic Object Detectors with Annotators Uncertainty

## Description
Abstract goes here

## Setup Environment
Our code extends the implementation of [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) and [probdet](https://github.com/asharakeh/probdet) (probabilistic Faster R-CNN and Retinanet).
1. Setup python environment following the installation instructions of each project (`YOLOX` and `probdet`).
2. Install required libraries for post-hoc calibration and model evaluation with (torch and torchvision compiled with cpu:
    ```bash
    pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
    ```
3. Download pretrained model weights for YOLOX and probdet (optional)
    ```bash
    cd YOLOX && mkdir pretrained_weights
    wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth -P pretrained_weights
    cd ..
    cd probdet && mkdir pretrained_weights
    # download deterministic and NLL for Faster R-CNN and Retinanet from gdrive
    ```
   
## Preparing Datasets
VOC-MIX dataset is readily available in `data` for this anonymized repository.
1. Create a new directory in `data` with your dataset name
2. Place the training, validation and test images into `train2017`, `val2017` and `test2017`, respectively
3. Prepare your annotations with YOLO (one txt file per image) format: x1,y1,x2,y2,class_id,annotator_id
4. Run the following script to preprocess them into expected COCO format
    ```bash
    python preprocess_data.py --yolo-txt-dirs <path to yolo annotations> --output-json data/<dataset name>/annotations/instances_{train/val/test}2017.json
    ```

## How to train
1. Follow command and training instructions of the subprojects:
   1. `YOLOX`: Create a new config file under `exps/default` and set 
      ```python
      self.uncertain = True
      self.bbox_unc_loss = 'dmm2' 
      ```
   2. `probdet`: Create a new config file under `src/configs/COCO-Detection` and set
      ```yaml
      PROBABILISTIC_MODELING:
          BBOX_COV_LOSS:
              NAME: 'dmmv2'
      ```
2. Perform inference on the hold-out dataset with the trained model, saving them in COCO json format.
3. Train the isotonic regression models with `python train_ir.py`

Examples:
```bash
```

## Evaluation
1. 