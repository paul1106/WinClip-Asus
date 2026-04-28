import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from datasets import *
from datasets import dataset_classes
from datasets.asus_visa import _load_meta, _instance_of, get_instance_gallery_map, ASUS_VISA_DIR
from utils.csv_utils import *
from utils.metrics import *
from utils.training_utils import *
from WinCLIP import *
from utils.eval_utils import *


def save_predictions(pred_dir, scores, gt_masks, gt_labels, img_paths, cls_name, specie_names, sample_ids):
    pred_dir = Path(pred_dir)
    pixel_dir = pred_dir / 'pixel_maps'
    gt_dir = pred_dir / 'gt_masks'
    pixel_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for idx in range(len(scores)):
        fname = f'{idx:08d}.npy'
        pixel_map = np.array(scores[idx], dtype=np.float32)
        gt_mask = (np.array(gt_masks[idx]) > 0).astype(np.uint8)

        np.save(pixel_dir / fname, pixel_map)
        np.save(gt_dir / fname, gt_mask)

        records.append({
            'index': idx,
            'sample_id': sample_ids[idx],
            'cls_name': cls_name,
            'specie_name': specie_names[idx],
            'anomaly': int(gt_labels[idx]),
            'img_path': img_paths[idx],
            'pred_score': float(pixel_map.max()),
            'pixel_map_path': f'pixel_maps/{fname}',
            'gt_mask_path': f'gt_masks/{fname}',
            'pixel_map_shape': list(pixel_map.shape),
        })

    with open(pred_dir / 'prediction_manifest.jsonl', 'w') as f:
        for r in records:
            f.write(json.dumps(r) + '\n')

    logger.info(f'Saved {len(records)} predictions to {pred_dir}')


def test(model,
         dataloader: DataLoader,
         device: str,
         is_vis: bool,
         img_dir: str,
         class_name: str,
         cal_pro: bool,
         train_data: DataLoader,
         resolution: int,
         pred_dir: str = None):

    model.eval_mode()

    logger.info('begin build text feature gallery...')
    model.build_text_feature_gallery(class_name)
    logger.info('build text feature gallery finished.')

    if train_data is not None:
        logger.info('begin build image feature gallery...')
        for (data, mask, label, name, img_type, _) in train_data:
            data = [model.transform(Image.fromarray(cv2.cvtColor(f.numpy(), cv2.COLOR_BGR2RGB))) for f in data]
            data = torch.stack(data, dim=0).to(device)
            model.build_image_feature_gallery(data)
        logger.info('build image feature gallery finished.')

    scores = []
    test_imgs = []
    gt_list = []
    gt_mask_list = []
    names = []
    img_paths = []
    specie_names = []

    for (data, mask, label, name, img_type, img_path) in dataloader:

        data = [model.transform(Image.fromarray(f.numpy())) for f in data]
        data = torch.stack(data, dim=0)

        for d, n, l, m, t, p in zip(data, name, label, mask, img_type, img_path):
            test_imgs += [denormalization(d.cpu().numpy())]
            l = l.numpy()
            m = m.numpy()
            m[m > 0] = 1

            names += [n]
            gt_list += [l]
            gt_mask_list += [m]
            img_paths += [p]
            specie_names += ['' if t in ('normal', 'good') else t]

        data = data.to(device)
        score = model(data)
        scores += score

    test_imgs, scores, gt_mask_list = specify_resolution(test_imgs, scores, gt_mask_list, resolution=(resolution, resolution))

    if is_vis:
        plot_sample_cv2(names, test_imgs, {'WinClip': scores}, gt_mask_list, save_folder=img_dir)

    if pred_dir is not None:
        sample_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_paths]
        save_predictions(pred_dir, scores, gt_mask_list, gt_list, img_paths, class_name, specie_names, sample_ids)
        return None

    return metric_cal(np.array(scores), gt_list, gt_mask_list, cal_pro=cal_pro)


def test_instance_shot(model, category, device, is_vis, img_dir, cal_pro,
                       gallery_map, resolution, batch_size, pred_dir=None):
    """Evaluate with per-instance 1-shot: each test instance gets its own gallery image."""
    model.eval_mode()
    model.build_text_feature_gallery(category)

    meta = _load_meta()
    root = ASUS_VISA_DIR
    test_entries = meta['test'].get(category, [])

    inst_entries = defaultdict(list)
    for e in test_entries:
        inst_entries[_instance_of(e['img_path'])].append(e)

    scores, test_imgs, gt_list, gt_mask_list, names = [], [], [], [], []
    img_paths, specie_names = [], []

    for inst_name, entries in inst_entries.items():
        gallery_path = gallery_map.get(inst_name)
        if gallery_path:
            gimg = cv2.imread(gallery_path, cv2.IMREAD_COLOR)
            gimg = cv2.resize(gimg, (1024, 1024))
            gimg_t = model.transform(Image.fromarray(gimg)).unsqueeze(0).to(device)
            model.build_image_feature_gallery(gimg_t)

        imgs_t = []
        for e in entries:
            abs_img_path = os.path.join(root, e['img_path'])
            img = cv2.imread(abs_img_path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (1024, 1024))
            img_t = model.transform(Image.fromarray(img))

            test_imgs.append(denormalization(img_t.numpy()))
            gt_list.append(e['anomaly'])
            names.append(f"{category}-{inst_name}-{os.path.splitext(os.path.basename(e['img_path']))[0]}")
            img_paths.append(abs_img_path)
            specie_names.append(e['specie_name'] if e['anomaly'] == 1 else '')

            if e['mask_path']:
                gt = cv2.imread(os.path.join(root, e['mask_path']), cv2.IMREAD_GRAYSCALE)
                gt = cv2.resize(gt, (1024, 1024), interpolation=cv2.INTER_NEAREST)
                gt[gt > 0] = 255
            else:
                gt = np.zeros((1024, 1024), dtype=np.uint8)
            gt_mask_list.append(gt)
            imgs_t.append(img_t)

        for i in range(0, len(imgs_t), batch_size):
            batch = torch.stack(imgs_t[i:i + batch_size]).to(device)
            scores += model(batch)

    test_imgs, scores, gt_mask_list = specify_resolution(
        test_imgs, scores, gt_mask_list, resolution=(resolution, resolution))

    if is_vis:
        plot_sample_cv2(names, test_imgs, {'WinClip': scores}, gt_mask_list, save_folder=img_dir)

    if pred_dir is not None:
        sample_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_paths]
        save_predictions(pred_dir, scores, gt_mask_list, gt_list, img_paths, category, specie_names, sample_ids)
        return None

    return metric_cal(np.array(scores), gt_list, gt_mask_list, cal_pro=cal_pro)


def main(args):
    kwargs = vars(args)

    logger.info('==========running parameters=============')
    for k, v in kwargs.items():
        logger.info(f'{k}: {v}')
    logger.info('=========================================')

    seeds = [111, 333, 999]
    kwargs['seed'] = seeds[kwargs['experiment_indx']]
    setup_seed(kwargs['seed'])

    if kwargs['use_cpu'] == 0:
        device = f"cuda:0"
    else:
        device = f"cpu"
    kwargs['device'] = device

    # prepare the experiment dir
    model_dir, img_dir, logger_dir, model_name, csv_path = get_dir_from_args(**kwargs)

    kwargs['out_size_h'] = kwargs['resolution']
    kwargs['out_size_w'] = kwargs['resolution']

    # get the model
    model = WinClipAD(**kwargs)
    model = model.to(device)

    shot_mode = kwargs.get('shot_mode', 'component')

    pred_dir = kwargs.get('pred_dir')

    if shot_mode == 'instance':
        gallery_map = get_instance_gallery_map(kwargs['class_name'], kwargs['experiment_indx'])
        metrics = test_instance_shot(
            model, kwargs['class_name'], device,
            is_vis=kwargs['vis'], img_dir=img_dir, cal_pro=kwargs['cal_pro'],
            gallery_map=gallery_map, resolution=kwargs['resolution'],
            batch_size=kwargs['batch_size'], pred_dir=pred_dir,
        )
    else:
        if kwargs['k_shot'] > 0:
            train_dataloader, _ = get_dataloader_from_args(phase='train', perturbed=False, **kwargs)
        else:
            train_dataloader = None

        test_dataloader, _ = get_dataloader_from_args(phase='test', perturbed=False, **kwargs)

        metrics = test(model, test_dataloader, device, is_vis=kwargs['vis'], img_dir=img_dir,
                       class_name=kwargs['class_name'], cal_pro=kwargs['cal_pro'],
                       train_data=train_dataloader, resolution=kwargs['resolution'],
                       pred_dir=pred_dir)

    if metrics is not None:
        logger.info(f"\n")
        for k, v in metrics.items():
            logger.info(f"{kwargs['class_name']}======={k}: {v:.2f}")
        save_metric(metrics, dataset_classes[kwargs['dataset']], kwargs['class_name'],
                    kwargs['dataset'], csv_path)
    else:
        logger.info(f'Skipped metrics. Predictions saved to {pred_dir}')


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def get_args():
    parser = argparse.ArgumentParser(description='Anomaly detection')
    parser.add_argument('--dataset', type=str, default='visa', choices=['mvtec', 'visa', 'asus_visa'])
    parser.add_argument('--class-name', type=str, default='candle')

    parser.add_argument('--img-resize', type=int, default=240)
    parser.add_argument('--img-cropsize', type=int, default=240)
    parser.add_argument('--resolution', type=int, default=400)

    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--vis', type=str2bool, choices=[True, False], default=True)
    parser.add_argument("--root-dir", type=str, default="./result_winclip")
    parser.add_argument("--load-memory", type=str2bool, default=True)
    parser.add_argument("--cal-pro", type=str2bool, default=False)
    parser.add_argument("--experiment_indx", type=int, default=0)
    parser.add_argument("--gpu-id", type=int, default=0)

    # pure test
    parser.add_argument("--pure-test", type=str2bool, default=False)

    # method related parameters
    parser.add_argument('--k-shot', type=int, default=0)
    parser.add_argument('--scales', nargs='+', type=int, default=(2, 3, ))
    parser.add_argument("--backbone", type=str, default="ViT-B-16-plus-240",
                        choices=['ViT-B-16-plus-240'])
    parser.add_argument("--pretrained_dataset", type=str, default="laion400m_e32")

    parser.add_argument("--use-cpu", type=int, default=0)
    parser.add_argument("--pred-dir", type=str, default=None,
                        help="If set, save predictions (pixel maps, gt masks, manifest) to this "
                             "directory and skip metric computation.")
    parser.add_argument("--shot-mode", type=str, default='component',
                        choices=['component', 'instance'],
                        help="1-shot selection strategy for asus_visa: "
                             "'component' picks one random normal from the class train split; "
                             "'instance' picks the closest train normal per test instance.")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    import os

    args = get_args()
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['CUDA_VISIBLE_DEVICES'] = f"{args.gpu_id}"
    main(args)
