import argparse
import os
import json
import cv2
import numpy as np
from src.util import read_raw_anns, preprocess_anns, convert_unc_bbox

# CHANGE CLASS NAME HERE IF NEEDED
CLASS_NAMES = ['aeroplane', 'bicycle', 'bird', 'boat',
                'bottle', 'bus', 'car', 'cat', 'chair',
                'cow', 'diningtable', 'dog', 'horse',
                'motorbike', 'person', 'pottedplant',
                'sheep', 'sofa', 'train', 'tvmonitor']

def main(opt):
    assert len(CLASS_NAMES) == opt.n_class, 'invalid number of classes'
    os.makedirs(os.path.dirname(opt.output_json), exist_ok=True)
    d = {'categories': [{'id': 0, 'name': 'bg'}] + [{'id': i, 'name': CLASS_NAMES[i - 1]} for i in
                                                    range(1, len(CLASS_NAMES) + 1)], 'images': [], 'annotations': []}

    all_files = os.listdir(opt.yolo_txt_dir)
    print(f'processing {len(all_files)} files')
    for f in all_files:
        if f.endswith('.txt'):
            img = cv2.imread(os.path.join(opt.image_dir, f[:-4]))
            height, width, _ = img.shape
            img_id = len(d['images']) + (10 ** len(str(len(all_files))))
            d['images'].append({'id': img_id, 'width': width, 'height': height, 'file_name': f[:-4]})

            anns = read_raw_anns(os.path.join(opt.yolo_txt_dir, f))
            unique_annid = np.unique(anns[:, 5])
            processed_anns = preprocess_anns(anns, len(unique_annid), opt.n_class)
            processed_anns = convert_unc_bbox(processed_anns, to_coco=True, n_class=opt.n_class+1)
            for ann in processed_anns:
                ann_id = len(d['annotations'])
                ann, raw_bbox = np.asarray(ann[:4 + opt.n_class+1]), ann[4 + opt.n_class+1:]
                assert len(raw_bbox) % 4 == 0
                d['annotations'].append({'id': ann_id, 'image_id': img_id, 'bbox': ann[:4].tolist(), 'iscrowd': 0,
                                         'category_id': int(ann[4:].argmax()), 'area': int(ann[2] * ann[3]),
                                         'class_logits': ann[4:].tolist(), 'raw_bbox': raw_bbox})

    with open(opt.output_json, 'w') as f:
        json.dump(d, f)
    print(f'saved json to {opt.output_json}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-txt-dir', type=str, help='path to yolo text annotations directory')
    parser.add_argument('--output-json', type=str, default='./data/dataset_name/annotations/instances_train2017.json', help='output json file path')
    parser.add_argument('--image-dir', type=str, default='./data/dataset_name/train2017', help='images path')
    parser.add_argument('--n-class', type=int, default=20, help='number of classes')

    opt = parser.parse_args()
    main(opt)
