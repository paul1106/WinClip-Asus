import os
import json
import random
from difflib import SequenceMatcher

ASUS_VISA_DIR = '/project/aimm_new/IAD_Dataset/ASUS_SMT/Unified_classes/Asus-Visa'

# All classes present in the test split
asus_visa_classes = [
    'BGA', 'C', 'C-R', 'CAE', 'CHOKE', 'CNET', 'IC', 'LED',
    'MELF', 'MLD', 'OSC', 'OTHER', 'QFN', 'R', 'R-L', 'R-R',
    'R0201', 'RNET', 'SOD', 'SOT23', 'TO',
]


def _load_meta():
    with open(os.path.join(ASUS_VISA_DIR, 'meta.json')) as f:
        return json.load(f)


def _instance_of(img_path):
    """Extract instance name from a relative img_path like 'BGA/J13_1@.../normal/000000.jpg'."""
    return img_path.split('/')[1]


def _find_closest_instance(target, candidates, rng):
    """
    Find the closest train instance to a test instance by name similarity.
    Priority: exact match > same ref_des > same board_code > sequence similarity.
    rng must be a seeded random.Random instance to ensure reproducibility.
    """
    if target in candidates:
        return target

    target_parts = target.split('@')
    ref_des = target_parts[0]
    board_code = target_parts[1] if len(target_parts) > 1 else ''

    same_ref = [c for c in candidates if c.split('@')[0] == ref_des]
    if same_ref:
        return rng.choice(same_ref)

    if board_code:
        same_board = [c for c in candidates
                      if len(c.split('@')) > 1 and c.split('@')[1] == board_code]
        if same_board:
            return rng.choice(same_board)

    return max(candidates, key=lambda c: SequenceMatcher(None, target, c).ratio())


def load_asus_visa(category, k_shot, experiment_indx):
    """
    Standard dataset loader (component-level k-shot).
    For k_shot=0: zero-shot, no train images returned.
    For k_shot>0: picks k_shot random normal images from the class train split.
    """
    assert category in asus_visa_classes
    assert k_shot in [0, 1, 5, 10]

    meta = _load_meta()
    root = ASUS_VISA_DIR

    # Test data
    test_entries = meta['test'].get(category, [])
    test_imgs, test_gts, test_labels, test_types = [], [], [], []
    for e in test_entries:
        img_abs = os.path.join(root, e['img_path'])
        mask_abs = os.path.join(root, e['mask_path']) if e['mask_path'] else 0
        test_imgs.append(img_abs)
        test_gts.append(mask_abs)
        test_labels.append(e['anomaly'])
        test_types.append(e['specie_name'] if e['anomaly'] == 1 else 'normal')

    # Train data (component-level selection)
    train_imgs, train_gts, train_labels, train_types = [], [], [], []
    if k_shot > 0:
        train_entries = meta['train'].get(category, [])
        normal_entries = [e for e in train_entries if e['anomaly'] == 0]
        rng = random.Random(42 + experiment_indx)
        selected = rng.sample(normal_entries, min(k_shot, len(normal_entries)))
        for e in selected:
            train_imgs.append(os.path.join(root, e['img_path']))
            train_gts.append(0)
            train_labels.append(0)
            train_types.append('normal')

    return (train_imgs, train_gts, train_labels, train_types), \
           (test_imgs, test_gts, test_labels, test_types)


def get_instance_gallery_map(category, experiment_indx=0):
    """
    Build a mapping {test_instance_name: abs_train_normal_image_path} for instance-level 1-shot.
    If a test instance has no train normal images, falls back to the closest train instance by name.
    """
    meta = _load_meta()
    root = ASUS_VISA_DIR

    train_entries = meta['train'].get(category, [])
    normal_train = [e for e in train_entries if e['anomaly'] == 0]

    train_inst_map = {}
    for e in normal_train:
        inst = _instance_of(e['img_path'])
        train_inst_map.setdefault(inst, []).append(e)

    if not train_inst_map:
        return {}

    test_entries = meta['test'].get(category, [])
    test_instances = sorted(set(_instance_of(e['img_path']) for e in test_entries))
    train_instances = list(train_inst_map.keys())

    rng = random.Random(42 + experiment_indx)
    gallery_map = {}
    for test_inst in test_instances:
        closest = _find_closest_instance(test_inst, train_instances, rng)
        entry = rng.choice(train_inst_map[closest])
        gallery_map[test_inst] = os.path.join(root, entry['img_path'])

    return gallery_map
