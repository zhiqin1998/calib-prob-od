import argparse
import json
import os
import pickle
import time

import numpy as np


def inference(model, json_input, save_path, loc_model=None, probdet=False):
    """
    Function to perform inference of the isotonic regression posthoc calibrator

    Parameters:
        model (dict[sklearn.isotonic.IsotonicRegression]): dict of sklearn isotonic regression models for class confidence, key is the class index
        json_input (str): path to prediction json file
        save_path (str): json file path to save calibrated predictions
        loc_model (dict[sklearn.isotonic.IsotonicRegression]): dict of sklearn isotonic regression models for box variance, key is 0, 1, 2, 3 for x1, y1, x2, y2
        probdet (bool): flag to indicate whether the prediction is from probdet or YOLOX, default is False (YOLOX)
    """
    with open(json_input, 'r') as f:
        preds = json.load(f)
    print(f'inferencing on {len(preds)} predictions')
    st = time.time()
    for i, pred in enumerate(preds):
        curr_cls_prob, curr_cls_id = pred['cls_prob'], pred['category_id']

        m = model.get(curr_cls_id + (1 if probdet else 0), None)
        if m is not None:
            new_score = np.clip(m.predict([curr_cls_prob[curr_cls_id]]), 0, 1)[0]
        else:
            new_score = curr_cls_prob[curr_cls_id]
        # paste new score and scale other class so that sum to 1
        curr_cls_prob = [x / (1 - curr_cls_prob[curr_cls_id]) * (1 - new_score) for x in curr_cls_prob]
        curr_cls_prob[curr_cls_id] = new_score
        pred['cls_prob'] = curr_cls_prob
        pred['score'] = curr_cls_prob[curr_cls_id]
        pred['category_id'] = curr_cls_id
        if loc_model is not None:
            if probdet:
                for j in range(len(pred['bbox_covar'])):
                    pred['bbox_covar'][j][j] = np.clip(loc_model[j].predict([pred['bbox_covar'][j][j]]), 0, None)[0]
            else:
                pred_box_var = pred['xyxy_bbox_var']
                for j, m in loc_model.items():
                    pred_box_var[j] = np.clip(m.predict([pred_box_var[j]]), 0, None)[0]
                pred['xyxy_bbox_var'] = pred_box_var
        preds[i] = pred

    print('done in', time.time() - st)
    with open(save_path, 'w') as f:
        json.dump(preds, f)
    print(f'saved results to {save_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred-json', type=str, help='path of prediction json file')
    parser.add_argument('--out-json', type=str, default='./outputs/dataset_name/calibrated_pred.json',
                        help='path of output json file')
    parser.add_argument('--class-model', type=str, default='/outputs/dataset_name/class_ir_model.pkl',
                        help='model path of class ir model')
    parser.add_argument('--loc-model', type=str, default='/outputs/dataset_name/loc_ir_model.pkl',
                        help='model path of loc ir model')
    parser.add_argument('--probdet', action='store_true', default=False,
                        help='prediction is from probdet project')

    opt = parser.parse_args()
    with open(opt.class_model, 'rb') as f:
        class_model = pickle.load(f)
    if os.path.isfile(opt.loc_model):
        with open(opt.loc_model, 'rb') as f:
            loc_model = pickle.load(f)
    else:
        loc_model = None
    os.makedirs(os.path.dirname(opt.out_json), exist_ok=True)

    inference(class_model, opt.pred_json, opt.out_json, loc_model=loc_model, probdet=opt.probdet)
