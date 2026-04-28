import argparse
import os

from loguru import logger

from datasets import dataset_classes
from eval_WinCLIP import eval_one_class
from utils.training_utils import setup_seed
from WinCLIP import WinClipAD

DATASET_SEED_BASE = 42   # dataset selection seed = DATASET_SEED_BASE + experiment_indx


def get_args():
    parser = argparse.ArgumentParser(description='Batch evaluation over all classes (model loaded once)')
    parser.add_argument('--dataset', type=str, default='asus_visa',
                        choices=['mvtec', 'visa', 'asus_visa'])
    parser.add_argument('--k-shot', type=int, default=0, choices=[0, 1, 5, 10])
    parser.add_argument('--shot-mode', type=str, default='component',
                        choices=['component', 'instance'],
                        help='1-shot strategy (only relevant when k-shot > 0): '
                             'component = 1 random normal from class train set; '
                             'instance = closest train instance per test instance')
    parser.add_argument('--pred-dir', type=str, default=None,
                        help='Base output directory. Each class is saved under '
                             '{pred-dir}[_seed{N}]/{class}. '
                             'If not set, metrics are computed instead.')
    parser.add_argument('--experiment-indx', type=int, default=0, choices=[0, 1, 2],
                        help='Controls random seed for k-shot selection (seed = 42 + experiment-indx)')
    parser.add_argument('--img-resize', type=int, default=240)
    parser.add_argument('--img-cropsize', type=int, default=240)
    parser.add_argument('--resolution', type=int, default=400)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--backbone', type=str, default='ViT-B-16-plus-240',
                        choices=['ViT-B-16-plus-240'])
    parser.add_argument('--pretrained-dataset', type=str, default='laion400m_e32')
    parser.add_argument('--scales', nargs='+', type=int, default=(2, 3))
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--use-cpu', type=int, default=0)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    os.environ['CURL_CA_BUNDLE'] = ''
    # Only set CUDA_VISIBLE_DEVICES if not already set by the caller
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    seeds = [111, 333, 999]
    setup_seed(seeds[args.experiment_indx])

    device = 'cpu' if args.use_cpu else 'cuda:0'

    print(f'[run_winclip] dataset={args.dataset}  k_shot={args.k_shot}  '
          f'shot_mode={args.shot_mode}  device={device}  '
          f'CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")}',
          flush=True)
    print('[run_winclip] Loading model (this may take ~30s)...', flush=True)
    model = WinClipAD(
        out_size_h=args.resolution,
        out_size_w=args.resolution,
        device=device,
        backbone=args.backbone,
        pretrained_dataset=args.pretrained_dataset,
        scales=args.scales,
        img_resize=args.img_resize,
        img_cropsize=args.img_cropsize,
    )
    model = model.to(device)
    print('[run_winclip] Model loaded.', flush=True)

    # Shared kwargs passed to eval_one_class for each class
    eval_kwargs = {
        'dataset': args.dataset,
        'k_shot': args.k_shot,
        'shot_mode': args.shot_mode,
        'experiment_indx': args.experiment_indx,
        'batch_size': args.batch_size,
        'resolution': args.resolution,
        'img_resize': args.img_resize,
        'img_cropsize': args.img_cropsize,
        'vis': False,
        'cal_pro': False,
    }

    # Append seed suffix to pred_dir when k_shot > 0
    pred_dir_base = args.pred_dir
    if pred_dir_base is not None and args.k_shot > 0:
        seed = DATASET_SEED_BASE + args.experiment_indx
        pred_dir_base = f'{pred_dir_base}_seed{seed}'

    classes = dataset_classes[args.dataset]
    for i, cls in enumerate(classes, 1):
        print(f'[run_winclip] [{i}/{len(classes)}] {cls}', flush=True)
        pred_dir = f'{pred_dir_base}/{cls}' if pred_dir_base else None
        eval_one_class(model, device, cls, eval_kwargs, pred_dir=pred_dir)

    print('[run_winclip] All classes done.', flush=True)
