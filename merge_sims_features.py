import argparse
import os
import re
import numpy as np


def detect_n_batch_n_group(folder_path):
    """Infer n_batch and n_group from saved text_emb_*_gt_b*.npy filenames."""
    pattern = re.compile(r'^text_emb_(\d+)_gt_b(\d+)\.npy$')
    max_group, max_batch = -1, -1
    for fname in os.listdir(folder_path):
        m = pattern.match(fname)
        if m:
            max_group = max(max_group, int(m.group(1)))
            max_batch = max(max_batch, int(m.group(2)))
    if max_batch == -1 or max_group == -1:
        raise FileNotFoundError(
            'No text_emb_*_gt_b*.npy files found in %s. '
            'Run feature extraction first.' % folder_path
        )
    return max_batch + 1, max_group + 1


def merge_sims(folder_path, n_batch):
    sims_output, sims_gt = None, None
    for batch_idx in range(n_batch):
        tmp_output = np.diag(np.load(os.path.join(folder_path, 'sims_output_%s.npy' % batch_idx)))
        tmp_gt = np.diag(np.load(os.path.join(folder_path, 'sims_gt_%s.npy' % batch_idx)))
        if batch_idx == 0:
            sims_output = tmp_output
            sims_gt = tmp_gt
        else:
            sims_output = np.concatenate((sims_output, tmp_output))
            sims_gt = np.concatenate((sims_gt, tmp_gt))

    np.save(os.path.join(folder_path, 'sims_gt.npy'), sims_gt)
    np.save(os.path.join(folder_path, 'sims_output.npy'), sims_output)
    print('Saved merged sims to %s' % folder_path)


def merge_features(folder_path, n_batch, n_group):
    img_emb, gt_emb, output_emb = None, None, None
    for batch_idx in range(n_batch):
        for idx in range(n_group):
            gt_path = os.path.join(folder_path, 'text_emb_%s_gt_b%s.npy' % (idx, batch_idx))
            output_path = os.path.join(folder_path, 'text_emb_%s_output_b%s.npy' % (idx, batch_idx))
            img_path = os.path.join(folder_path, 'img_emb_%s_gt_b%s.npy' % (idx, batch_idx))
            # Last batch may have fewer shards than earlier ones — skip missing files
            if not (os.path.exists(gt_path) and os.path.exists(output_path) and os.path.exists(img_path)):
                continue
            tmp_gt = np.load(gt_path)
            tmp_output = np.load(output_path)
            tmp_img = np.load(img_path)
            if img_emb is None:
                img_emb = tmp_img
                gt_emb = tmp_gt
                output_emb = tmp_output
            else:
                img_emb = np.concatenate((img_emb, tmp_img))
                gt_emb = np.concatenate((gt_emb, tmp_gt))
                output_emb = np.concatenate((output_emb, tmp_output))

    if img_emb is None:
        raise FileNotFoundError('No feature shards found in %s' % folder_path)

    np.save(os.path.join(folder_path, 'img_emb.npy'), img_emb)
    np.save(os.path.join(folder_path, 'gt_emb.npy'), gt_emb)
    np.save(os.path.join(folder_path, 'output_emb.npy'), output_emb)
    print('Saved merged features to %s (gt=%d, output=%d, img=%d)' % (
        folder_path, len(gt_emb), len(output_emb), len(img_emb)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge per-batch feature and similarity files.')
    parser.add_argument('--features_base_dir', type=str, required=True,
                        help='Base directory where feature files are saved')
    parser.add_argument('--feature_folder', type=str, required=True,
                        help='Sub-folder name inside features_base_dir')
    parser.add_argument('--n_batch', type=int, default=None,
                        help='Number of batches (auto-detected from saved files if not provided)')
    parser.add_argument('--n_group', type=int, default=None,
                        help='Number of shards/groups per batch (auto-detected from saved files if not provided)')
    parser.add_argument('--merge_sims', action='store_true', default=False,
                        help='Merge similarity matrix files')
    parser.add_argument('--merge_features', action='store_true', default=False,
                        help='Merge feature embedding files')
    args = parser.parse_args()

    folder_path = os.path.join(args.features_base_dir, args.feature_folder)

    n_batch, n_group = args.n_batch, args.n_group
    if n_batch is None or n_group is None:
        detected_n_batch, detected_n_group = detect_n_batch_n_group(folder_path)
        n_batch = n_batch if n_batch is not None else detected_n_batch
        n_group = n_group if n_group is not None else detected_n_group
        print('Auto-detected: n_batch=%d, n_group=%d' % (n_batch, n_group))

    if args.merge_sims:
        merge_sims(folder_path, n_batch)
    if args.merge_features:
        merge_features(folder_path, n_batch, n_group)
