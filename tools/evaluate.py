import numpy as np
import os
import sys

import fire
import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)


def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]


def _draw(X, Ys, title=None, xlabel=None, ylabel=None, output_image_path='result.png'):
    for Y in Ys:
        plt.plot(X, Y, label=Y.name)
    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)

    plt.legend()
    plt.savefig(output_image_path, bbox_inches='tight')


def evaluate(label_path,
             result_path,
             label_split_file,
             current_class=0,
             coco=False,
             score_thresh=-1):
    import lib.datasets.kitti.kitti_eval_python.eval as eval
    import lib.datasets.kitti.kitti_eval_python.kitti_common as kitti

    dt_annos = kitti.get_label_annos(result_path)
    if score_thresh > 0:
        dt_annos = kitti.filter_annos_low_score(dt_annos, score_thresh)
    val_image_ids = _read_imageset_file(label_split_file)
    gt_annos = kitti.get_label_annos(label_path, val_image_ids)
    if coco:
        return eval.get_coco_eval_result(gt_annos, dt_annos, current_class)
    else:
        return eval.get_official_eval_result(gt_annos, dt_annos, current_class)


def multi_eval(label_path,
               result_path,
               label_split_file,
               current_class=0,
               coco=False,
               score_thresh=-1,
               depth_min=0,
               depth_max=60,
               depth_step=10,
               output_csv_path='result.csv'):
    import lib.datasets.kitti.kitti_eval_python.eval as eval
    import lib.datasets.kitti.kitti_eval_python.kitti_common as kitti

    def get_filter_fn(lower_bound, upper_bound):
        return lambda x: lower_bound < float(x[13]) <= upper_bound

    result = []
    for depth in range(depth_min, depth_max, depth_step):
        print('=' * 10, f'depth = {depth} ~ {depth + depth_step}', '=' * 10)
        # x[13] is the depth coordinate of the bbox
        filter_fn = get_filter_fn(depth, depth + depth_step)

        dt_annos = kitti.get_label_annos(result_path, filter_fn=filter_fn)
        if score_thresh > 0:
            dt_annos = kitti.filter_annos_low_score(dt_annos, score_thresh)
        val_image_ids = _read_imageset_file(label_split_file)
        gt_annos = kitti.get_label_annos(label_path, val_image_ids, filter_fn=filter_fn)

        if coco:
            ret_str, ret_dict, _ = eval.get_coco_eval_result(gt_annos, dt_annos, current_class)
        else:
            ret_str, ret_dict, _ = eval.get_official_eval_result(gt_annos, dt_annos, current_class)
        print(ret_str)
        ret_dict['depth'] = depth
        result.append(ret_dict)

    if output_csv_path:
        df = pd.DataFrame.from_records(result)
        df.to_csv(output_csv_path, index=False)


def draw_from_df(df, x, ys, title=None, xlabel=None, ylabel=None, output_image_path='result.png'):
    if isinstance(df, str):
        df = pd.read_csv(df)
    if xlabel is None:
        xlabel = x
    if ylabel is None:
        ylabel = ys[0]
    shift = int((df[x][1] - df[x][0]) / 2)
    _draw(df[x] + shift, [df[y] for y in ys], title=title, xlabel=xlabel, ylabel=ylabel, output_image_path=output_image_path)


def draw_num_samples_of_depth(label_path,
                              label_split_file,
                              depth_min=0,
                              depth_max=60,
                              depth_step=10,
                              output_image_path='kitti_num_samples.png'):
    import lib.datasets.kitti.kitti_eval_python.kitti_common as kitti

    val_image_ids = _read_imageset_file(label_split_file)
    gt_annos = kitti.get_label_annos(label_path, val_image_ids)
    gt_annos_depth = np.concatenate([anno['location'][:, 2] for anno in gt_annos])

    gt_annos_depth = gt_annos_depth[(gt_annos_depth <= depth_max) & (gt_annos_depth >= 0)]

    print(f'num_gt_annos: {len(gt_annos)}, num_bboxes: {len(gt_annos_depth)}')

    hist, _, _ = plt.hist(gt_annos_depth,
                          bins=int((depth_max - depth_min) / depth_step),
                          range=(depth_min, depth_max),
                          weights=np.ones_like(gt_annos_depth) / len(gt_annos_depth))

    plt.title('Proportion of bbox sample')
    plt.savefig(output_image_path, bbox_inches='tight')


if __name__ == '__main__':
    fire.Fire()
