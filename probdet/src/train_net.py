"""
Probabilistic Detectron Training Script following Detectron2 training script found at detectron2/tools.
"""
import core
import os
import sys

# This is very ugly. Essential for now but should be fixed.
sys.path.append(os.path.join(core.top_dir(), 'src', 'detr'))

# Detectron imports
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader, build_detection_train_loader, DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer, launch
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators, verify_results
from detectron2.data.datasets import register_coco_instances, load_coco_json

# DETR imports
from d2.train_net import Trainer as Detr_Trainer

# Project imports
from core.setup import setup_config, setup_arg_parser
from custom_dataset import UncDatasetMapper

def custom_register_coco_instances(name, metadata, json_file, image_root):
    """
    Register a dataset in COCO's json annotation format for
    instance detection, instance segmentation and keypoint detection.
    (i.e., Type 1 and 2 in http://cocodataset.org/#format-data.
    `instances*.json` and `person_keypoints*.json` in the dataset).

    This is an example of how to register a new dataset.
    You can do something similar to this function, to register new datasets.

    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root, name, extra_annotation_keys=['raw_bbox', 'class_logits']))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="coco", **metadata
    )

register_coco_instances('vocmix_val', {}, '../data/voc_mix/processed_coco/annotations/instances_val2017.json', '../data/voc_mix/processed_coco/val2017')
register_coco_instances('vocmix_test', {}, '../data/voc_mix/processed_coco/annotations/instances_test2017.json', '../data/voc_mix/processed_coco/test2017')

register_coco_instances('vocmix_train_ind', {}, '../data/voc_mix/processed_coco_unc_ind/annotations/instances_train2017.json', '../data/voc_mix/processed_coco_unc_ind/train2017')

custom_register_coco_instances('vocmix_train_unc', {}, '../data/voc_mix/processed_coco_unc/annotations/instances_train2017.json', '../data/voc_mix/processed_coco_unc/train2017')


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        """
        Builds evaluators for post-training mAP report.
        Args:
            cfg(CfgNode): a detectron2 CfgNode
            dataset_name(str): registered dataset name

        Returns:
            detectron2 DatasetEvaluators object
        """
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluators = [COCOEvaluator(dataset_name, cfg, True, output_folder)]
        return DatasetEvaluators(evaluators)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Builds DataLoader for test set.
        Args:
            cfg(CfgNode): a detectron2 CfgNode
            dataset_name(str): registered dataset name

        Returns:
            detectron2 DataLoader object specific to the test set.
        """
        return build_detection_test_loader(
            cfg, dataset_name)

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Builds DataLoader for train set.
        Args:
            cfg(CfgNode): a detectron2 CfgNode

        Returns:
            detectron2 DataLoader object specific to the train set.
        """
        if 'ANN_UNCERTAINTY' in cfg and cfg.ANN_UNCERTAINTY:
            mapper = UncDatasetMapper(cfg, True)
        else:
            mapper = None
        return build_detection_train_loader(
            cfg, mapper=mapper)


def main(args):
    # Setup config node
    cfg = setup_config(args,
                       random_seed=args.random_seed)

    # For debugging only
    # cfg.defrost()
    # cfg.DATALOADER.NUM_WORKERS = 0
    # cfg.SOLVER.IMS_PER_BATCH = 1

    # Eval only mode to produce mAP results
    # Build Trainer from config node. Begin Training.
    if cfg.MODEL.META_ARCHITECTURE == 'ProbabilisticDetr':
        trainer = Detr_Trainer(cfg)
    else:
        trainer = Trainer(cfg)

    if args.eval_only:
        model = trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    # Create arg parser
    arg_parser = setup_arg_parser()

    args = arg_parser.parse_args()
    print("Command Line Args:", args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
