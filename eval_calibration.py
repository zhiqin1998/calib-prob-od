import argparse

import numpy as np
from pycocotools.coco import COCO
from src.util import load_predictions, compute_metrics


def main(opt):
    # load gt and pred
    print('loading gt and pred')
    cocoGt = COCO(opt.gt_json)
    n_class = len(cocoGt.cats)
    filename_to_imgid = {}
    for img in cocoGt.dataset['images']:
        filename_to_imgid[img['file_name']] = img['id']
    # load predictions
    preds, all_files = load_predictions(opt.pred_json, opt.gt_json)
    # load preprocessed annotation clusters (gt)
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

    unc_metrics, _ = compute_metrics(preds[0], preds[1], preds[2], all_gts, all_raw_bboxes, iou_thres=0.5, )
    print(f'TVD: \t\t{unc_metrics[0]:.5f}\n'
          f'TVD (FP):\t{unc_metrics[1]:.5f}\n'
          f'LUE: \t\t{unc_metrics[2]:.5f}\n'
          f'FNE: \t\t{unc_metrics[3]:.5f}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt-json', type=str, help='path of ground truth json file')
    parser.add_argument('--pred-json', type=str, help='path of prediction json file')

    opt = parser.parse_args()
    main(opt)
