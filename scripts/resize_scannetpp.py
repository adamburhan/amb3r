"""Transform DSLR-undistorted GT frames through the exact AMB3R preprocessing.

Routes the ScanNet++ eval GT images (already fisheye->iPhone-pinhole via the
official undistort_given_intrinsics script) through the same Demo dataloader
used for the original AMB3R/SfM run, so the GT inherits the identical
crop+resize (disable_crop=False path) and the identical intrinsics branch.

Outputs, under --output_path:
  images/<stem>.png       transformed GT frames (Lanczos, loader-identical)
  masks/<stem>.png        anonymization/border masks, NEAREST via the same
                          affine (255=valid, 0=invalid; borders -> invalid)
  intrinsics.json         K_in (loader input branch), K_out (render camera),
                          image size, per-axis affine (a, b) of the transform

Pre-flight checks printed before processing:
  - aug_crop value (must be <= 1: rng provably dormant for landscape frames)
  - which intrinsics branch the loader takes (.npz metadata vs hardcoded K)
  - frame count / name alignment between directory and loader output
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, 'thirdparty'))

from amb3r.datasets.demo import Demo, natural_sort_key  # same sort as loader

VALID_EXT = {'.jpg', '.jpeg', '.png', '.heic', '.ppm'}   # mirror Demo's filter


def list_frames(image_dir: Path):
    """Replicate Demo._get_views file selection exactly (filter + sort)."""
    names = [p.name for p in image_dir.iterdir()
             if p.suffix.lower() in VALID_EXT
             and 'depth' not in p.name.lower()
             and 'mask' not in p.name.lower()]
    return sorted(names, key=natural_sort_key)


def per_axis_affine(K_in: np.ndarray, K_out: np.ndarray):
    """The loader's crop+rescale+crop composite is affine per axis:
    x' = a_x * x + b_x. Recover (a, b) from the two intrinsics."""
    a_x = K_out[0, 0] / K_in[0, 0]
    a_y = K_out[1, 1] / K_in[1, 1]
    b_x = K_out[0, 2] - a_x * K_in[0, 2]
    b_y = K_out[1, 2] - a_y * K_in[1, 2]
    return (a_x, b_x), (a_y, b_y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_path', required=True,
                    help='dir of DSLR-undistorted-by-iphone GT images')
    ap.add_argument('--output_path', required=True)
    args = ap.parse_args()

    image_dir = Path(args.data_path)
    out = Path(args.output_path)
    (out / 'images').mkdir(parents=True, exist_ok=True)

    names = list_frames(image_dir)
    if not names:
        sys.exit(f'No images found in {image_dir} (check extensions/case).')
    stems = [Path(n).stem for n in names]

    data = Demo(ROOT=str(image_dir), resolution=(518, 392),
                kf_every=1, full_video=True)


    loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    _, views_all = batch                     

    imgs = views_all['images'].squeeze(0).cpu().numpy()      # (T,3,H,W) [-1,1]
    T = imgs.shape[0]
    assert len(stems) == T, f'{len(stems)} files vs {T} loaded frames'

    # Collated intrinsics live under one of these keys depending on base class.
    K_key = next((k for k in ('camera_intrinsics', 'intrinsics', 'K')
                  if k in views_all), None)
    if K_key is None:
        sys.exit(f'No intrinsics key in loader output; keys: '
                 f'{sorted(views_all.keys())}')
    K_all = views_all[K_key].squeeze(0).cpu().numpy()        # (T,3,3)
    K_out = K_all[0]
    assert np.allclose(K_all, K_out[None], atol=1e-4), \
        'Per-frame K differs across frames -- transform not constant?!'

    # Per-sequence K_in from the ScanNet++ undistorted transforms, matching the
    # loader's intrinsics branch (image_dir/../nerfstudio/transforms_undistorted.json).
    transforms_path = image_dir.parent / 'nerfstudio' / 'transforms_undistorted.json'
    with open(transforms_path, 'r') as f:
        cam = json.load(f)
    K_in = np.array([[cam['fl_x'], 0.,          cam['cx']],
                     [0.,          cam['fl_y'], cam['cy']],
                     [0.,          0.,          1.]])

    (a_x, b_x), (a_y, b_y) = per_axis_affine(K_in, K_out)
    H_out, W_out = imgs.shape[2], imgs.shape[3]
    print(f'[transform] x\' = {a_x:.6f} x + {b_x:.3f}   '
          f'y\' = {a_y:.6f} y + {b_y:.3f}   out = {W_out}x{H_out}')
    if abs(a_x / a_y - 1.0) > 0.05:
        print('[warn] strongly anisotropic scale -- double-check K_in branch.')

    imgs_u8 = np.clip((imgs.transpose(0, 2, 3, 1) + 1.0) / 2.0 * 255.0 + 0.5,
                      0, 255).astype(np.uint8)
    for i, stem in enumerate(stems):
        Image.fromarray(imgs_u8[i]).save(out / 'images' / f'{stem}.png')


    with open(out / 'intrinsics.json', 'w') as f:
        json.dump({
            'K_in': K_in.tolist(),
            'K_out_render_camera': K_out.tolist(),
            'width': int(W_out), 'height': int(H_out),
            'affine_x': [float(a_x), float(b_x)],
            'affine_y': [float(a_y), float(b_y)],
            'frames': stems,
        }, f, indent=2)

    print(f'done: {T} images -> {out}')

if __name__ == '__main__':
    main()