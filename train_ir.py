import argparse
import copy
import os
import pickle

import numpy as np
from scipy import stats
from pycocotools.coco import COCO
from sklearn.isotonic import IsotonicRegression

from src.util import get_dt_gt_match, load_predictions


def isotonic_regression(dataset, use_weight=False, n_sample=10, loc=False):
    """
    Helper function to train isotonic regression models on the given dataset

    Parameters:
        dataset (dict[tuple]): a dict with tuples of training input, training target and sample weights
        use_weight (bool): whether to use sample weights for isotonic regression
        n_sample (int): minimum number of samples
        loc (bool): flag to indicate whether the output range is unbounded (variance can range from 0 - inf), default is False

    Returns:
        dict[sklearn.isotonic.IsotonicRegression]: dict of isotonic regression models, same keys as in dataset
    """
    # train isotonic regression model
    model = {}
    for cls_id, data in dataset.items():
        scores, labels, weights = data
        if len(scores) < n_sample:
            model[cls_id] = None
        else:
            model[cls_id] = IsotonicRegression(y_min=0., y_max=None if loc else 1., out_of_bounds='clip').fit(scores,
                                                                                                              labels,
                                                                                                              weights if use_weight else None)
    return model


def batch_dt_gt_match(box_preds, box_vars, cls_preds, gts, raw_gt_bboxes, iou_thres=0.1, pred_xyxy=True, max_det=100):
    """
    Batch version of the get_dt_gt_match function

    Parameters:
        box_preds (list[list[float]]): list of n length containing the predicted boxes mean in xyxy format
        box_vars (list[list[float]]): list of n length containing the predicted boxes variances in xyxy format
        cls_preds (list[list[float]]): list of n predicted class probabilities with shape num_class+1
        gts (list[list[float]]): list of n length containing the gt boxes and class probs in xyxy + (num_class+1) format
        raw_gt_bboxes (list[list[float]]): list of n length containing the raw gt boxes in xyxy format
        iou_thres (float): IoU threshold
        pred_xyxy(bool): flag to indicate whether boxes are in xyxy format or not, default is True
        max_det (int): maximum number of detections

    Returns:
        list[tuple]: n results from get_dt_gt_match function
        list[list[float]]: updated list of box_preds
        list[list[float]]: updated list of box_vars
        list[list[float]]: updated list of cls_preds
        list[list[float]]: updated list of n gts
        list[list[float]]: updated list of n raw_gt_bboxes
    """
    # batch matching of prediction to gt
    assert len(box_preds) == len(box_vars) == len(cls_preds) == len(gts) == len(raw_gt_bboxes)
    box_preds, box_vars, cls_preds, gts, raw_gt_bboxes = copy.deepcopy(box_preds), copy.deepcopy(
        box_vars), copy.deepcopy(cls_preds), copy.deepcopy(gts), copy.deepcopy(raw_gt_bboxes)
    ret = []
    updated_box_preds, updated_box_vars, updated_cls_preds = [], [], []
    updated_gts, updated_raw_gt_bboxes = [], []

    for box_pred, box_var, cls_pred, gt, raw_gt_bbox in zip(box_preds, box_vars, cls_preds, gts, raw_gt_bboxes):
        # sort dt by conf
        dt_sorted_idx = cls_pred[:, 0].argsort()  # smallest to largest bg prob
        # sorted_idx = cls_pred[:, 1:].max(1).argsort()[::-1]  # largest to smallest class prob
        box_pred, box_var, cls_pred = box_pred[dt_sorted_idx], box_var[dt_sorted_idx], cls_pred[dt_sorted_idx]
        if max_det is not None:
            box_pred, box_var, cls_pred = box_pred[:max_det], box_var[:max_det], cls_pred[:max_det]
        # sort gt by obj
        gt_sorted_idx = gt[:, 4].argsort()  # smallest to largest bg prob
        gt, raw_gt_bbox = gt[gt_sorted_idx], [raw_gt_bbox[i] for i in gt_sorted_idx]
        if pred_xyxy:
            _box_pred = box_pred
        else:
            _box_pred = box_pred.copy()
            _box_pred[:, 2] = _box_pred[:, 0] + _box_pred[:, 2]
            _box_pred[:, 3] = _box_pred[:, 1] + _box_pred[:, 3]
        gt_match_inds, dt_match_inds = get_dt_gt_match(gt[:, :4], _box_pred, iou_thres)
        ret.append([gt_match_inds, dt_match_inds])
        updated_box_preds.append(_box_pred)
        updated_box_vars.append(box_var)
        updated_cls_preds.append(cls_pred)
        updated_gts.append(gt)
        updated_raw_gt_bboxes.append(raw_gt_bbox)
    return ret, (updated_box_preds, updated_box_vars, updated_cls_preds), updated_gts, updated_raw_gt_bboxes


def prepare_train_labels(all_matches, cls_preds, gts, ignore_bg=False, fp_weight_scale=10):
    """
    Convert the prediction-gt match from batch_dt_gt_match function into a suitable format for sklearn. For class
    confidence calibration.

    Parameters:
        all_matches (list[tuple]): n results from get_dt_gt_match function
        cls_preds (list[list[float]]): list of predicted class probabilities with shape num_class+1
        gts (list[list[float]]): list of n length containing the gt boxes and class probs in xyxy + (num_class+1) format
        ignore_bg (bool): flag to indicate whether to remove bg class or not, default is False
        fp_weight_scale (float): scale of weights for fp, default is 10, set to None to disable

    Returns:
        dict[tuple]: a dict with tuples of training input, training target and sample weights
    """
    dataset = {}
    for all_match, cls_pred, gt in zip(all_matches, cls_preds, gts):
        gt_match_inds, dt_match_inds = all_match
        if ignore_bg:
            cls_pred = copy.deepcopy(cls_pred)
            obj_score = 1 - cls_pred[:, 0:1]
            cls_pred[:, 1:] = cls_pred[:, 1:] / obj_score
            gt = copy.deepcopy(gt)
            obj_score = 1 - gt[:, 4:5]
            gt[:, 5:] = gt[:, 5:] / obj_score
        for i, gind in enumerate(dt_match_inds):
            obj_score = 1 - cls_pred[i, 0]
            if gind < 0 and fp_weight_scale is not None:
                obj_score = obj_score / fp_weight_scale  # fp is roughly 10x tp

            cls_id = cls_pred[i, 1:].argmax() + 1
            score = cls_pred[i, cls_id]
            if gind > -1:
                agreement = gt[gind, 4 + cls_id]
            else:
                agreement = 0
            if cls_id not in dataset:
                dataset[cls_id] = [[], [], []]
            dataset[cls_id][0].append(score)
            dataset[cls_id][1].append(agreement)
            dataset[cls_id][2].append(obj_score)

    return dataset


def prepare_train_labels_loc(all_matches, box_vars, cls_preds, gts, raw_gt_bboxes, exclude_fp=True):
    """
    Convert the prediction-gt match from batch_dt_gt_match function into a suitable format for sklearn. For bounding
    box variance calibration.

    Parameters:
        all_matches (list[tuple]): n results from get_dt_gt_match function
        box_vars (list[list[float]]): list of n length containing the predicted boxes variances in xyxy format
        cls_preds (list[list[float]]): list of n predicted class probabilities with shape num_class+1
        gts (list[list[float]]): list of n length containing the gt boxes and class probs in xyxy + (num_class+1) format
        raw_gt_bboxes (list[list[float]]): list of n length containing the raw gt boxes in xyxy format
        exclude_fp (bool): flag to exclude fp or not

    Returns:
        dict[tuple]: a dict with tuples of training input, training target and sample weights
    """
    dataset = {}
    for all_match, box_var_pred, cls_pred, gt, raw_bbox in zip(all_matches, box_vars, cls_preds, gts, raw_gt_bboxes):
        gt_match_inds, dt_match_inds = all_match
        for i, gind in enumerate(dt_match_inds):
            obj_score = 1 - cls_pred[i, 0]
            if gind < 0 and exclude_fp:
                continue
            matched_pred_var = box_var_pred[i]
            alpha = gt[gind, 4]
            matched_raw_box = np.asarray(raw_bbox[gind]).reshape((-1, 4))
            if len(matched_raw_box) > 1:
                error = np.maximum((matched_raw_box.max(0) - matched_raw_box.mean(0)),
                                   (matched_raw_box.mean(0) - matched_raw_box.min(0)))
                error = error + np.maximum(error * 0.1, 2)  # relax
                z = stats.norm.ppf(1 - alpha / 2)
                target_var = (error / z) ** 2
                for j in range(4):
                    if j not in dataset:
                        dataset[j] = [[], [], []]
                    dataset[j][0].append(matched_pred_var[j])
                    dataset[j][1].append(target_var[j])
                    dataset[j][2].append(obj_score)
    return dataset


def main(opt):
    # load gt and pred
    print('loading gt and pred')
    cocoGt = COCO(opt.gt_json)
    n_class = len(cocoGt.cats)
    filename_to_imgid = {}
    for img in cocoGt.dataset['images']:
        filename_to_imgid[img['file_name']] = img['id']
    preds, all_files = load_predictions(opt.pred_json, opt.gt_json)
    all_gts, all_raw_bboxes = [], []
    for f in all_files:
        img_id = filename_to_imgid[f]
        coco_anns = cocoGt.imgToAnns[img_id]
        if len(coco_anns):
            anns, raw_bboxes = [], []
            for ann in coco_anns:
                bbox = ann['bbox']
                bbox[2] = bbox[0] + bbox[2]
                bbox[3] = bbox[1] + bbox[3]
                anns.append(bbox + ann['class_logits'])
                raw_bboxes.append(ann['raw_bbox'])
            gts = np.asarray(anns)
        else:
            gts, raw_bboxes = np.zeros((0, 4 + n_class + 1)), []
        all_gts.append(gts)
        all_raw_bboxes.append(raw_bboxes)

    # match gt to pred
    all_matches, sorted_preds, sorted_gts, sorted_raw_bboxes = batch_dt_gt_match(preds[0], preds[1], preds[2], all_gts,
                                                                                 all_raw_bboxes)
    print(f'training isotonic regression models with sample weight={opt.use_weight}')
    dataset = prepare_train_labels(all_matches, sorted_preds[2], sorted_gts)
    ir_model = isotonic_regression(dataset, use_weight=opt.use_weight)

    loc_dataset = prepare_train_labels_loc(all_matches, sorted_preds[1], sorted_preds[2], sorted_gts, sorted_raw_bboxes)
    loc_ir_model = isotonic_regression(loc_dataset, use_weight=opt.use_weight, loc=True)

    os.makedirs(opt.out_dir, exist_ok=True)
    with open(os.path.join(opt.out_dir, 'class_ir_model.pkl'), 'wb') as f:
        pickle.dump(ir_model, f)

    with open(os.path.join(opt.out_dir, 'loc_ir_model.pkl'), 'wb') as f:
        pickle.dump(loc_ir_model, f)

    print(f'saved class_ir_model and loc_ir_model to {opt.out_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt-json', type=str, help='path of ground truth json file')
    parser.add_argument('--pred-json', type=str, help='path of prediction json file')
    parser.add_argument('--out-dir', type=str, default='./outputs/dataset_name/', help='output directory to save model')
    parser.add_argument('--use-weight', type=bool, default=True, help='use weight')

    opt = parser.parse_args()
    main(opt)
