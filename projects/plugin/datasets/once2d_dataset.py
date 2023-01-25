import os
from time import sleep
import numpy as np
import mmcv
from mmcv import ProgressBar
import subprocess
import re
import datetime
import time
import threading
import json
from tqdm import tqdm
import cv2

from mmdet.datasets.builder import DATASETS
from .tusimple_dataset import TuSimpleDataset
from tools.ganet.curvelane.curvelane_evaluate import LaneMetricCore

@DATASETS.register_module()
class once2dDataset(TuSimpleDataset):

    def __init__(self,
                 data_root,
                 data_list,
                 pipeline,
                 test_mode=False,
                 test_suffix='png',
                 work_dir=None,
                 **kwargs
        ):
        super(once2dDataset, self).__init__(
            data_root,
            data_list,
            pipeline,
            test_mode,
            test_suffix,
            work_dir,
            **kwargs
        )
        self.evaluator = LaneMetricCore(
                eval_width=224,
                eval_height=224,
                iou_thresh=0.5,
                lane_width=5
            )
        self.evaluate_data_list = data_list
    
    def set_ori_shape(self, idx):
        ori_filename = self.img_infos[idx]['raw_file']
        filename = os.path.join(self.img_prefix, ori_filename)
        img_tmp = cv2.imread(filename)
        ori_shape = img_tmp.shape
        self.img_infos[idx]['ori_shape']=ori_shape
    
    def set_all_scenes(self):
        pass

    # 重载函数
    def parser_datalist(self, data_list):
        img_infos = []
        #print(data_list)
        for anno_list in data_list:
            with open(anno_list) as f:
                lines = f.readlines()
                for line in lines:
                    raw_file = line.strip()
                    raw_file = raw_file[1:] if raw_file[0]=='/' else raw_file
                    img_info = dict(raw_file=os.path.join('img', raw_file))
                    # if self.test_mode == False: # 不论模式都加载anno
                    raw_anno_file = raw_file.replace('.jpg', '.json')
                    img_info.update(dict(anno_file=os.path.join('all_gt', raw_anno_file)))
                    img_infos.append(img_info)
        #print(img_info)
        return img_infos

    def _set_group_flag(self):
        self.flag = np.zeros(len(self), dtype=np.uint8) # 0
        for i in range(len(self)):
            self.flag[i] = 1

    def __len__(self):
        return len(self.img_infos)

    def load_labels(self, idx, offset_x, offset_y):
        anno_file = self.img_prefix + "/" + self.img_infos[idx]['anno_file']
        lanes = []
        #print(anno_file)
        with open(anno_file, 'r') as anno_f:
            #lines = anno_f.readlines() # [],会有空的
            lines = json.load(anno_f)
        for lane in lines['lanes']:
            coords = []
            for x, y in lane:
                coords.append(float(x) + offset_x)
                coords.append(float(y) + offset_y)
            if len(coords) > 3:
                lanes.append(coords)
        id_classes = [1 for i in range(len(lanes))]
        id_instances = [i + 1 for i in range(len(lanes))]
        return lanes, id_classes, id_instances

    def format_results(self, outputs, **kwargs):
        save_dir = self.work_dir + '/format'
        bar = ProgressBar(len(outputs))
        for idx, output in enumerate(outputs):
            result, virtual_center, cluster_center = output['final_dict_list']
            culane_lanes = culane_convert_formal(result)
            save_file = save_dir + '/' + output['img_metas']['ori_filename'].replace('.jpg', '.lines.txt')
            mmcv.mkdir_or_exist(os.path.dirname(save_file))
            f_pr = open(save_file,'w')
            for lane in culane_lanes:
                for v in lane:
                    f_pr.write(str(v)+' ')
                f_pr.write('\n')
            bar.update()

        f_pr.close()
        print(f"\nwriting culane results to {save_dir}")
        return save_dir

    # Use the framework in curvelane_dataset.py to perform evaluation on the model
    def evaluate(self, outputs, **eval_kwargs):
        self.evaluator.reset()
        # format和culane完全一样
        pr_dir = self.format_results(outputs) if outputs else self.work_dir + '/format'
        gt_dir = self.data_root
        for idx in tqdm(range(len(self.img_infos))):
            raw_file = self.img_infos[idx]['raw_file']
            pr_anno = os.path.join(pr_dir,raw_file) 
            gt_anno = os.path.join(gt_dir,raw_file) 
            pr = parse_anno(pr_anno)
            gt = parse_anno_gt(gt_anno.replace('img', 'all_gt'))
            if 'ori_shape' not in self.img_infos[idx]:
                self.set_ori_shape(idx)
            ori_shape = self.img_infos[idx]['ori_shape']
            gt_wh = dict(height=ori_shape[0], width=ori_shape[1]) # (1440, 2560, 3)
            predict_spec = dict(Lines=pr, Shape=gt_wh) # gt_wh：{'height': 1440, 'width': 2560}
            target_spec = dict(Lines=gt, Shape=gt_wh)
            self.evaluator(target_spec, predict_spec)

        metric = self.evaluator.summary()
        if self.check_or_not: 
            z_metric = self.check(pr_dir)
            metric.update(**z_metric)
        return metric


def culane_convert_formal(lanes):
    res = []
    for lane in lanes:
        lane_coords = []
        sety = set()
        for coord in lane:
            x = round(coord[0])
            y = round(coord[1])
            if y not in sety:
                lane_coords.append(x)
                lane_coords.append(y)
                sety.add(y)
        res.append(lane_coords)
    return res

def convert_coords_formal(lanes):
    res = []
    for lane in lanes:
        lane_coords = []
        for coord in lane:
            lane_coords.append({'x': coord[0], 'y': coord[1]})
        res.append(lane_coords)
    return res

def parse_anno(filename, formal=True):
    anno_dir = filename.replace('.jpg', '.lines.txt') 
    annos = []
    with open(anno_dir, 'r') as anno_f:
        lines = anno_f.readlines()
    for line in lines: # '734.02 1439.0 815.89 1366.21 897.76 1293.43 979.62 1220.64 1061.49 1147.86 1061.86 1147.53 1178.79 1046.65 1254.7 973.04 1284.6 920.13 1307.6 890.22 1452.51 812.01 \n'
        coords = []
        numbers = line.strip().split(' ')
        coords_tmp = [float(n) for n in numbers]

        for i in range(len(coords_tmp) // 2):
            coords.append((coords_tmp[2 * i], coords_tmp[2 * i + 1]))
        annos.append(coords)
    if formal: # true
        annos = convert_coords_formal(annos)
    return annos

def parse_anno_gt(filename, formal=True):
    anno_dir = filename.replace('.jpg', '.json') 
    annos = []
    with open(anno_dir, 'r') as anno_f:
            #lines = anno_f.readlines() # [],会有空的
        lines = json.load(anno_f)
    for lane in lines['lanes']:
        coords = []
        for x, y in lane:
            coords.append((float(x), float(y)))
        if len(coords) > 3:
            annos.append(coords)
    if formal: # true
        annos = convert_coords_formal(annos)
    return annos
