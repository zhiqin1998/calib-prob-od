import copy
import json
from collections import defaultdict

import torch
import torchvision
import numpy as np
from scipy import stats
from pycocotools.coco import COCO


def read_raw_anns(ann_file):
    """
    Function to read raw annotators annotations with format x1, y1, x2, y2, cls_id, ann_id

    Parameters:
        ann_file (str): the annotation file path

    Returns:
        np.ndarray: nx6 loaded annotations with format x1, y1, x2, y2, cls_id, ann_id
    """
    anns = []
    with open(ann_file, 'r') as f:
        for line in f.readlines():
            x1, y1, x2, y2, cls_id, ann_id = list(map(int, line.split(',')))
            anns.append([x1, y1, x2, y2, cls_id, ann_id])
    if len(anns):
        return np.array(anns)
    else:
        return np.zeros((0, 6))


def convert_unc_bbox(proc_ann, to_coco=True, n_class=None):
    """
    Helper function to convert list of bounding boxes from xyxy to xywh and vice versa

    Parameters:
        proc_ann (list[float], np.ndarray): nxm raw annotation with format *box coordinates, *class_probs, *raw_boxes
        to_coco (bool): boolean flag to convert xyxy to xywh, default is true
        n_class (int): number of classes

    Returns:
        np.ndarray: nxm converted annotation with format *box coordinates, *class_probs, *raw_boxes
    """
    assert n_class is not None
    proc_ann = copy.deepcopy(proc_ann)

    def box_ops(ii, jj):
        if to_coco:
            proc_ann[ii][jj] = proc_ann[ii][jj] - proc_ann[ii][jj - 2]
        else:
            proc_ann[ii][jj] = proc_ann[ii][jj] + proc_ann[ii][jj - 2]

    for i in range(len(proc_ann)):
        for j in range(len(proc_ann[i])):
            k = j
            if 4 <= j < n_class + 4:
                continue
            if j >= n_class + 4:
                k = j - n_class - 4
            if (k % 4) in (2, 3):
                box_ops(i, j)
    return proc_ann


def bbox_iou(box1, box2, x1y1x2y2=True, eps=1e-7):
    """
    Returns the IoU of box1 to box2

    Parameters:
        box1 (np.ndarray): array of shape 4
        box2 (np.ndarray): array of shape nx4
        x1y1x2y2 (bool): boolean flag to indicate whether box is in x1y1x2y2 format
        eps (float): epsilon for numerical stability

    Returns:
        np.ndarray: IoU of box1 to box2, with length of n
    """
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = np.maximum(np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1), 0) * \
            np.maximum(np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1), 0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    return iou  # IoU


def preprocess_anns(anns, n_annotator, n_class, box_iou_thres=0.1):
    """
    Preprocessing function to cluster annotations and compute soft class probabilities

    Parameters:
        anns (list[int], np.ndarray): nx6 of annotations with format x1, y1, x2, y2, cls_id, ann_id
        n_annotator (int): number of annotators
        n_class (int): number of classes
        box_iou_thres (float): IoU threshold, default is 0.1

    Returns:
        list[float]: list of clustered annotations with format x1, y1, x2, y2, *class_probs, *raw_boxes
    """

    def find_and_cluster(labels):
        def group_box(groups, labels):
            curr_box, labels = labels[0], labels[1:]
            best_i = -1
            best_n = -1
            curr_ann_id = curr_box[-1]
            for i, group in enumerate(groups):
                if (group[:, -1] == curr_ann_id).any():  # group cannot have multiple labels of same annotator
                    continue
                match_count = (bbox_iou(curr_box[:4], group[:, :4]) > box_iou_thres).sum().item()
                match = match_count > int(len(group) / 2)  # consider matched if over half of box has iou > threshold
                if match and match_count > best_n:
                    best_i = i
                    best_n = match_count
            if best_i >= 0:
                groups[best_i] = np.concatenate((groups[best_i], np.expand_dims(curr_box, 0)), axis=0)
            else:
                groups.append(np.expand_dims(curr_box, 0))
            return groups, labels

        groups = []  # list of array of array
        while len(labels):
            groups, labels = group_box(groups, labels)
        # attempt to merge group if have distinct ann id and box iou > thres
        i = 0
        while i + 1 < len(groups):
            curr_anns = groups[i][:, -1]
            curr_box = groups[i][:, :4].mean(axis=0)
            found = False
            for j in range(i + 1, len(groups)):
                oth_anns = groups[j][:, -1]
                if not np.isin(curr_anns, oth_anns).any():
                    oth_box = groups[j][:, :4].mean(axis=0, keepdims=True)
                    if (bbox_iou(curr_box, oth_box) > box_iou_thres).all():
                        # merge and restart
                        # print('merging', groups[i], groups[j])
                        groups[i] = np.concatenate((groups[i], groups[j]), axis=0)
                        del groups[j]
                        i = 0
                        found = True
                if found:
                    break
            if not found:
                i += 1
        return groups

    def reduce_groups(groups, n_annotator):
        labels = np.zeros((len(groups), 4 + n_class + 1))  # +1 for bg
        keep = []
        for i, group in enumerate(groups):
            classes = np.concatenate((group[:, 4] + 1, np.full((n_annotator - len(group),), 0)))  # 0 is bg
            class_logits = np.eye(n_class + 1)[classes.astype(int)].mean(axis=0)

            final_box = np.zeros((4,))
            # for iou matching of cluster
            final_box[:4] = group[:, :4].mean(axis=0)

            assert final_box[2] > final_box[0] and final_box[3] > final_box[1]
            labels[i, 4:] = class_logits
            labels[i, :4] = final_box.round()
            keep.append(i)

        # return list instead of array, with raw bbox appended at end
        ret = []
        for i in keep:
            temp = labels[i, :4].astype(int).tolist() + labels[i, 4:].tolist()
            ret.append(temp + groups[i][:, :4].flatten().round().astype(int).tolist())
        return ret

    groups = find_and_cluster(anns)
    return reduce_groups(groups, n_annotator)


def load_predictions(pred_json, gt_json, gt_files=None, conf_thres=None, bg_thres=None):
    """
    Load json predictions from yolox and probdet model and convert to the same format for evaluation

    Parameters:
        pred_json (str): path to pred json file
        gt_json (str): path to gt json file (annotation clusters json)
        gt_files (list): list of filenames to consider, by default infer from gt_json
        conf_thres (float): confidence threshold
        bg_thres (float): background threshold

    Returns:
        list[list[float]]: list of n length containing the predicted boxes mean in xyxy format
        list[list[float]]: list of n length containing the predicted boxes variances in xyxy format
        list[list[float]]: list of n predicted class probabilities with shape num_class+1
        list[str]: list of n filenames corresponding to the index of the lists above
    """
    with open(pred_json, 'r') as f:
        preds = json.load(f)
    cocoGt = COCO(gt_json)
    if gt_files is None:
        gt_files = [x['file_name'] for x in cocoGt.dataset['images']]
    res_dict = defaultdict(list)
    # convert result to dict image_id: annotations
    n_class = len([x for x in cocoGt.cats if cocoGt.cats[x]['name'] != 'bg']) # remove bg
    assert n_class <= len(preds[0]['cls_prob']) <= n_class + 1
    if 'xyxy_bbox_var' not in preds[0] and 'bbox_covar' not in preds[0]:
        print('warning no box variance, ignore metrics from localisation')
    for pred in preds:
        if conf_thres is not None:
            if pred['score'] < conf_thres:
                continue
        if bg_thres is not None:
            if pred['cls_prob'][0] > bg_thres:
                continue
        pred = copy.deepcopy(pred)
        if isinstance(pred['image_id'], str):
            filename = pred['image_id']
        else:
            filename = cocoGt.imgs[pred['image_id']]['file_name']
        res_dict[filename].append(pred)
    pred_boxes, pred_boxes_var, cls_preds = [], [], []
    for f in gt_files:
        pred_box, pred_box_var, cls_pred = [], [], []
        for pred in res_dict[f]:
            bbox = pred['bbox']
            bbox[2] = bbox[0] + bbox[2]
            bbox[3] = bbox[1] + bbox[3]
            pred_box.append(bbox)
            if 'xyxy_bbox_var' in pred:  # yolox
                pred_box_var.append(pred['xyxy_bbox_var'])
                cls_pred.append(pred['cls_prob'])
            elif 'bbox_covar' in pred:  # probdet
                pred_box_var.append([x[i] for i, x in enumerate(pred['bbox_covar'])])
                if len(pred['cls_prob']) == n_class + 1:  # frcnn
                    cls_pred.append(pred['cls_prob'][-1:] + pred['cls_prob'][:-1])
                else:  # retinanet
                    bg_prob = 1 - max(pred['cls_prob'])
                    temp = [bg_prob] + copy.deepcopy(pred['cls_prob'])
                    tot = sum(temp)
                    cls_pred.append([x / tot for x in temp])
            else:
                raise NotImplementedError
        if len(pred_box):
            pred_box = np.asarray(pred_box)
            cls_pred = np.asarray(cls_pred)
            pred_box_var = np.asarray(pred_box_var)

            pred_boxes.append(pred_box)
            pred_boxes_var.append(pred_box_var)
            cls_preds.append(cls_pred)
        else:
            pred_boxes.append(np.zeros((0, 4)))
            pred_boxes_var.append(np.zeros((0, 4)))
            cls_preds.append(np.zeros((0, n_class + 1)))
    # pred_boxes is xyxy, pred_boxes_var is shape 4 for the variance, cls_preds is shape num_class+1 (bg conf, raw_classes_conf), should sum to 1
    # gt_files is the list of filename corresponding to the index of pred_boxes, pred_boxes_var, cls_preds
    return (pred_boxes, pred_boxes_var, cls_preds), gt_files


def get_dt_gt_match(gt_box, pred_box, iou_thres):
    """
    Matching function to match detection to ground truth similar to COCO

    Parameters:
         gt_box (list[float]): list of mx4 xyxy gt boxes
         pred_box (list[float]): list of nx4 xyxy predicted boxes
         iou_thres (float): IoU threshold

    Returns:
        list[int]: list of n indices corresponding to the index of the predicted boxes, -1 means no match
        list[int]: list of m indices corresponding to the index of the gt boxes, -1 means no match
    """
    ious = torchvision.ops.box_iou(torch.from_numpy(gt_box), torch.from_numpy(pred_box)).numpy()
    gt_match_inds, dt_match_inds = np.ones((len(gt_box),), dtype=int) * -1, np.ones((len(pred_box),), dtype=int) * -1
    # match each det to best gt with highest iou > iou_thres
    if len(gt_box):
        for i in range(len(pred_box)):
            curr_iou = ious[:, i].copy()
            curr_iou = np.where(gt_match_inds < 0, curr_iou, -1)  # skip already matched gts
            max_idx = curr_iou.argmax()
            if curr_iou[max_idx] >= iou_thres:
                gt_match_inds[max_idx] = i
                dt_match_inds[i] = max_idx
    return gt_match_inds, dt_match_inds


def compute_metrics(box_preds, box_vars, cls_preds, gts, raw_gt_bboxes, iou_thres=0.5, verbose=False, pred_xyxy=True,
                    max_det=100):
    """
    Function to compute the metrics with all n returned results of load_predictions function

    Parameters:
        box_preds (list[list[float]]): list of n length containing the predicted boxes mean in xyxy format
        box_vars (list[list[float]]): list of n length containing the predicted boxes variances in xyxy format
        cls_preds (list[list[float]]): list of n predicted class probabilities with shape num_class+1
        gts (list[list[float]]): list of n length containing the gt boxes and class probs in xyxy + (num_class+1) format
        raw_gt_bboxes (list[list[float]]): list of n length containing the raw gt boxes in xyxy format
        iou_thres (float): IoU threshold
        verbose (bool): flag to show processing steps for debugging, default is False
        pred_xyxy(bool): flag to indicate whether boxes are in xyxy format or not, default is True
        max_det (int): maximum number of detections

    Returns:
        float: total variation distance of tp and fn (range 0 to 1)
        float: total variation distance of fp (range 0 to 1)
        float: localization uncertainty error of tp (range 0 to 1)
        float: false negative error of fn (range 0 to 1)
        float: localization uncertainty error of fp (range 0 to 1), this is same as tvd of fp
        int: tp count
        int: fp count
        int: fn count
    """
    # perform matching of prediction to gt and compute calibration metrics
    assert len(box_preds) == len(box_vars) == len(cls_preds) == len(gts) == len(raw_gt_bboxes)
    box_preds, box_vars, cls_preds, gts, raw_gt_bboxes = copy.deepcopy(box_preds), copy.deepcopy(
        box_vars), copy.deepcopy(cls_preds), copy.deepcopy(gts), copy.deepcopy(raw_gt_bboxes)
    all_matched_tvds, all_fp_tvds = [], []
    all_matched_loc_score, all_fn_loc_score, all_fp_loc_score = [], [], []
    print_i = 0
    for box_pred, box_var, cls_pred, gt, raw_gt_bbox in zip(box_preds, box_vars, cls_preds, gts, raw_gt_bboxes):
        # sort dt by conf
        sorted_idx = cls_pred[:, 0].argsort()  # smallest to largest bg prob
        # sorted_idx = cls_pred[:, 1:].max(1).argsort()[::-1]  # largest to smallest class prob
        box_pred, box_var, cls_pred = box_pred[sorted_idx], box_var[sorted_idx], cls_pred[sorted_idx]
        if max_det is not None:
            box_pred, box_var, cls_pred = box_pred[:max_det], box_var[:max_det], cls_pred[:max_det]
        # sort gt by obj
        sorted_idx = gt[:, 4].argsort()  # smallest to largest bg prob
        gt, raw_gt_bbox = gt[sorted_idx], [raw_gt_bbox[i] for i in sorted_idx]
        if pred_xyxy:
            _box_pred = box_pred
        else:
            _box_pred = box_pred.copy()
            _box_pred[:, 2] = _box_pred[:, 0] + _box_pred[:, 2]
            _box_pred[:, 3] = _box_pred[:, 1] + _box_pred[:, 3]
        gt_match_inds, dt_match_inds = get_dt_gt_match(gt[:, :4], _box_pred, iou_thres)
        if verbose and print_i < 3:
            print_i += 1
            for i, dti in enumerate(gt_match_inds):
                if dti < 0:
                    print(f'no match for {gt[i]}')
                else:
                    print(f'gt {gt[i]} match to pred {box_pred[dti]}, {cls_pred[dti]}')

        matched_tvds, fp_tvds = calc_mean_tvd(cls_pred, gt[:, 4:], dt_match_inds, gt_match_inds,
                                              verbose=verbose and print_i <= 3)
        all_matched_tvds.extend(matched_tvds)
        all_fp_tvds.extend(fp_tvds)
        if len(raw_gt_bbox):
            n_anns = (len(raw_gt_bbox[0]) // 4) / (1 - gt[0, 4])  # n annotator for image
        else:
            n_anns = None  # doesnt matter
        matched_loc_scores, fn_loc_scores, fp_loc_scores = calc_loc_ci_score(box_pred, box_var, 1 - cls_pred[:, 0],
                                                                             raw_gt_bbox,
                                                                             dt_match_inds, gt_match_inds, n_anns,
                                                                             verbose=verbose and print_i <= 3,
                                                                             pred_xyxy=pred_xyxy)
        all_matched_loc_score.extend(matched_loc_scores)
        all_fn_loc_score.extend(fn_loc_scores)
        all_fp_loc_score.extend(fp_loc_scores)
    # return (tvd, tvd_fp, lue, fne, lue_fp {same as tvd_fp}), (tp_count, fp_count, fn_count)
    return tuple(np.mean(x) if len(x) else 0. for x in
                 (all_matched_tvds, all_fp_tvds, all_matched_loc_score, all_fn_loc_score, all_fp_loc_score)), (
        len(all_matched_loc_score), len(all_fp_loc_score), len(all_fn_loc_score))


def calc_mean_tvd(preds, gts, dt_match_inds, gt_match_inds, verbose=False):
    """
    Function to compute mean total variation distance of one data sample (image)

    Parameters:
        preds (list[float]): list of n predicted class probabilities of (n x num_class+1)
        gts (list[float]): list of m ground truth class probabilities of (m x num_class+1)
        dt_match_inds (list[int]): list of n gt indices that the prediction is matched to, -1 means no match
        gt_match_inds (list[int]): list of m prediction indices that the ground truth is matched to, -1 means no match
        verbose (bool): flag to show processing steps for debugging, default is False

    Returns:
        float: total variation distance of tp and fn (range 0 to 1)
        float: total variation distance of fp (range 0 to 1)
    """
    if verbose:
        print('calculate tvd')
    if len(preds) == 0 and len(gts) == 0:
        return [], []
    tvds = []
    bg_prob = np.zeros_like(preds[0]) if len(preds) else np.zeros_like(gts[0])
    bg_prob[0] = 1  # set bg to 1
    # ious = torchvision.ops.box_iou(torch.from_numpy(gts[:, :4]), torch.from_numpy(preds[:, :4])).numpy()
    # calculate tvd for each gt, for unmatched pred and gt -> assume missing is 100% bg prob
    for i, gt_match_ind in enumerate(gt_match_inds):
        if gt_match_ind >= 0:
            tvds.append(np.abs(gts[i] - preds[gt_match_ind]).sum() / 2)  # tp
        else:  # -1
            tvds.append(np.abs(gts[i] - bg_prob).sum() / 2)  # fn
    assert len(tvds) == len(gts)
    # tvds.extend([1. for _ in range(sum(pred_unmatched_mask))]) # fp
    fp_tvds = []
    for i, dt_match_ind in enumerate(dt_match_inds):
        if dt_match_ind == -1:
            fp_tvds.append(np.abs(preds[i] - bg_prob).sum() / 2)  # fp
    if verbose:
        print(tvds, fp_tvds)
    return tvds, fp_tvds


def chunker(seq, size):
    """Helper function to chunk a long list into equal size lists"""
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def calc_loc_ci_score(box_preds, var_preds, fg_scores, raw_gt_bboxes, dt_match_inds, gt_match_inds, n_anns,
                      verbose=False, pred_xyxy=True):
    """
    Function to compute localization uncertainty error of one data sample (image)

    Parameters:
        box_preds (list[float]): list of n predicted box mean of (n x 4)
        var_preds (list[float]): list of n predicted box variance of (n x 4)
        fg_scores (list[float]): list of n predicted certainty
        raw_gt_bboxes (list[list[float]]): list of m raw bounding boxes of all annotators
        dt_match_inds (list[int]): list of n gt indices that the prediction is matched to, -1 means no match
        gt_match_inds (list[int]): list of m prediction indices that the ground truth is matched to, -1 means no match
        n_anns (int): number of annotators
        verbose (bool): flag to show processing steps for debugging, default is False
        pred_xyxy (bool): flag to indicate whether boxes are in xyxy format or not, default is True

    Returns:
        float: localization uncertainty error of tp (range 0 to 1)
        float: false negative error of fn (range 0 to 1)
        float: localization uncertainty error of fp (range 0 to 1), this is same as tvd of fp
    """
    # preds is x1 y1 x2 y2 or xywh
    # raw_bboxes is n * (x1, y1, x2, y2)
    if verbose:
        print('calculate loc ci score')
    if len(box_preds) == 0 and len(raw_gt_bboxes) == 0:
        return [], [], []
    if not pred_xyxy:
        # raw_gt_bboxes = convert_unc_bbox(raw_gt_bboxes, to_coco=True, n_class=0, have_logits=False)# convert gt to xywh
        # convert variance back to xyxy
        for i in range(len(var_preds)):
            var_preds[i][2] = var_preds[i][2] - var_preds[i][0]
            var_preds[i][3] = var_preds[i][3] - var_preds[i][1]
    scores = []
    fn_scores = []
    for i, gt_match_ind in enumerate(gt_match_inds):
        if gt_match_ind >= 0:
            lower_bound, upper_bound = stats.norm.interval(min(fg_scores[gt_match_ind], 0.999),
                                                           loc=box_preds[gt_match_ind], scale=np.sqrt(
                    var_preds[gt_match_ind]))  # prevent inf with min
            if verbose:
                print(fg_scores[gt_match_ind], lower_bound, upper_bound, raw_gt_bboxes[i])
                print([int(((lower_bound < raw_gt_box) & (raw_gt_box < upper_bound)).all()) for raw_gt_box in
                       chunker(raw_gt_bboxes[i], 4)])
            perc_gt_in_interval = sum(
                [int(((lower_bound < raw_gt_box) & (raw_gt_box < upper_bound)).all()) for raw_gt_box in
                 chunker(raw_gt_bboxes[i], 4)]) / n_anns  # percentage of annotated bbox in interval
            true_perc = (len(raw_gt_bboxes[i]) // 4) / n_anns
            pred_perc = fg_scores[gt_match_ind]
            if verbose:
                print(perc_gt_in_interval, true_perc, pred_perc)
            scores.append(
                abs(perc_gt_in_interval - pred_perc))  # diff between predicted fg confidence and actual number of annotated bbox in interval
        else:  # -1
            fn_scores.append(
                abs((len(raw_gt_bboxes[i]) // 4) / n_anns))  # penalise for not predicting when there are annotated bbox

    fp_scores = []
    for i, dt_match_ind in enumerate(dt_match_inds):
        if dt_match_ind == -1:
            fp_scores.append(abs(fg_scores[i]))  # fp
    if verbose:
        print(scores, fn_scores, fp_scores)
    return scores, fn_scores, fp_scores
