# Calibration of Probabilistic Object Detectors with Annotators Uncertainty

## Description
High degrees of disagreement among annotators can exist for ambiguous objects, e.g. in medical images, underscoring the challenges of establishing ground truth annotations in object detection tasks. Despite this, all existing object detectors implicitly require access to ground truth annotations for either training or evaluation. 
The fundamental questions we target are: How can we learn an object detector with multiple annotators' annotations but without objective ground truth annotations due to object ambiguity, and how can we enable the learned detector to express meaningful model predictive uncertainties in detecting ambiguous objects? 
To answer these questions, we present a highly interpretable approach to calibrate probabilistic object detectors, where the calibration goal is to align the class confidence and bounding box variance estimates to the annotators' annotation distribution. 
We introduce an efficient yet effective framework to calibrate probabilistic object detectors by designing four evaluation metrics to measure calibration errors regarding classification and localization, and proposing a train-time calibration and post-hoc calibrator, all without the need to access any ground truth. This framework is model-agnostic as it can be adapted to any existing probabilistic object detectors. Empirical results with real-world and synthetic datasets of medical and natural images demonstrate the superior performance of the proposed framework with three popular object detectors. 

## Setup Environment
Our code extends the implementation of [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) and [probdet](https://github.com/asharakeh/probdet) (probabilistic Faster R-CNN and Retinanet).
1. Setup python environment following the installation instructions of each project (`YOLOX` and `probdet`).
2. Install required libraries for post-hoc calibration and model evaluation with (remove torch and torchvision from requirements.txt if already installed):
    ```bash
    pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
    ```
3. Download pretrained model weights for YOLOX and probdet (optional)
    ```bash
    cd YOLOX && mkdir pretrained_weights
    wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth -P pretrained_weights
    cd ..
    cd probdet && mkdir pretrained_weights
    # download weights of deterministic and NLL for Faster R-CNN and Retinanet from gdrive (check original repo readme.md)
    ```
   
## Preparing Datasets
VBD-CXR and VOC-MIX datasets are readily available in `data` for this anonymized repository. 
Follow the steps below to bring your own dataset:
1. Create a new directory in `data` with your dataset name
2. Place the training, validation and test images into `train2017`, `val2017` and `test2017`, respectively
3. Prepare your annotations with YOLO (one txt file per image) format: x1,y1,x2,y2,class_id,annotator_id
4. Run the following script to cluster raw annotations and preprocess each set into expected COCO format
    ```bash
    python preprocess_data.py --yolo-txt-dir <path to yolo annotations> --output-json data/<dataset name>/annotations/instances_{train/val/test}2017.json \
                --image-dir data/<dataset name>/{train/val/test}2017/ --n-class <number of classes>
    ```

## How to train
1. Follow training instructions and commands of the subprojects by first creating their respective config file:
   1. `YOLOX`: Create a new config file under `exps/default` and set 
      ```python
      self.uncertain = True
      self.bbox_unc_loss = 'dmm' 
      ```
   2. `probdet`: Create a new config file under `src/configs/COCO-Detection` and set
      ```yaml
      PROBABILISTIC_MODELING:
          BBOX_COV_LOSS:
              NAME: 'dmmv2'
      ```
2. Perform inference on the hold-out dataset with the trained object detector, saving them in COCO json format.
3. Train the isotonic regression models with `python train_ir.py`

Example commands for training and calibrate probabilistic YOLOX on VOC-MIX dataset:
```bash
cd YOLOX
python tools/train.py -f exps/default/yolox_l_vocmix_uncertain.py -d 1 -b 16 --fp16 -o -c pretrained_weights/yolox_l.pth
python tools/eval.py -f exps/default/yolox_l_vocmix_uncertain.py -d 1 -b 16 --fp16 --ckpt <ckpt_file> --save-path YOLOX_outputs/vocmix_val_pred.json
cd ..
python train_ir.py --gt-json data/vocmix/annotations/instances_val2017.json --pred-json YOLOX/YOLOX_outputs/vocmix_val_pred.json --out-dir outputs/vocmix/
```

## Inference and Evaluation
1. Perform inference on the test dataset with the trained object detector, saving them in COCO json format
2. Perform inference with the trained isotonic regression models with `python test_ir.py`
3. Compute calibration evaluation metric with `python eval_calibration.py`

Example commands for inferencing and evaluating probabilistic YOLOX on VOC-MIX test dataset:
```bash
cd YOLOX
python tools/eval.py -f exps/default/yolox_l_vocmix_uncertain.py -d 1 -b 16 --fp16 --ckpt <ckpt_file> --test --save-path YOLOX_outputs/vocmix_test_pred.json
cd ..
python test_ir.py --pred-json YOLOX/YOLOX_outputs/vocmix_test_pred.json --out-json outputs/vocmix/ir_calibrated_test.json \
           --class-model outputs/vocmix/class_ir_model.pkl  --loc-model outputs/vocmix/loc_ir_model.pkl   
python eval_calibration.py --gt-json data/vocmix/annotations/instances_test2017.json --pred-json outputs/vocmix/ir_calibrated_test.json
```

Additionally, to compute other metrics such as LRP and PDQ if ground truth is available, please use the original repository at [LRP-Error](https://github.com/kemaloksuz/LRP-Error) and [pdq_evaluation](https://github.com/david2611/pdq_evaluation).
