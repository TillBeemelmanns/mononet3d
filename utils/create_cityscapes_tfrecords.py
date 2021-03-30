import os
import sys

sys.path.append("/src")

import glob
import cv2 as cv
import json
import numpy as np
import tensorflow as tf
import pyquaternion

from utils import tf_utils, helpers, kitti_utils
from utils import box3dImageTransform as box_utils


def create_example(img, scan, label):
    feature = {
        'image/img': tf_utils.bytes_feature(img),
        'image/orig': tf_utils.float_list_feature(label['orig'].reshape(-1, 1)),
        'image/calib': tf_utils.float_list_feature(label['calib'].reshape(-1, 1)),
        'scan/points': tf_utils.float_list_feature(scan[:, :3].reshape(-1, 1)),
        'label/clf': tf_utils.int64_list_feature(label['clf'].reshape(-1, 1)),
        'label/c_3d': tf_utils.float_list_feature(label['c_3d'].reshape(-1, 1)),
        'label/bbox_3d': tf_utils.float_list_feature(label['bbox_3d'].reshape(-1, 1)),
        'label/c_2d': tf_utils.float_list_feature(label['c_2d'].reshape(-1, 1)),
        'label/bbox_2d': tf_utils.float_list_feature(label['bbox_2d'].reshape(-1, 1)),
        'label/extent': tf_utils.float_list_feature(label['extent'].reshape(-1, 1)),
        'label/rotation_i': tf_utils.float_list_feature(label['ri'].reshape(-1, 1)),
        'label/rotation_j': tf_utils.float_list_feature(label['rj'].reshape(-1, 1)),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


CLASS_MAP = {
    'car': 0,

    'truck': 1,
    'bus': 1,
    'on rails': 1,
    'train': 1,
    'motorcycle': 1,
    'bicycle': 1,
    'caravan': 1,
    'trailer': 1,
    'dynamic': 1,
    'tunnel': 1,
    'ignore': 1
}


def create_records():
    max_objects = cfg['max_objects']

    for dataset, dataset_out in zip(cfg['datasets'], cfg['datasets_out']):

        img_dir = os.path.join(cfg['in_dir'], 'leftImg8bit', dataset)
        label_dir = os.path.join(cfg['in_dir'], 'gtBbox3d', dataset)

        img_files = glob.glob(os.path.join(img_dir, '*/*.png'))
        img_files = sorted(img_files)

        label_files = glob.glob(os.path.join(label_dir, '*/*.json'))
        label_files = sorted(label_files)

        n_scenes = cfg['n_scenes'] if cfg['n_scenes'] > 0 else len(img_files)
        bar = helpers.progbar(n_scenes)
        bar.start()

        with tf.io.TFRecordWriter(dataset_out) as writer:

            for scene_id, (img_file, label_file) in enumerate(zip(img_files, label_files)):

                if scene_id == n_scenes: break

                bar.update(scene_id)

                assert os.path.basename(img_file)[:-16] == os.path.basename(label_file)[:-14]

                img_arr = cv.imread(img_file)
                orig_img_size = img_arr.shape[:2]
                img_arr = cv.resize(img_arr, (cfg['img_size'][1], cfg['img_size'][0]), interpolation=cv.INTER_CUBIC)
                _, img = cv.imencode('.png', img_arr)
                img = img.tobytes()

                with open(label_file) as json_file:
                    label_dict = json.load(json_file)

                if len(label_dict['objects']) == 0: continue

                camera = box_utils.Camera(fx=label_dict['sensor']['fx'],
                                          fy=label_dict['sensor']['fy'],
                                          u0=label_dict['sensor']['u0'],
                                          v0=label_dict['sensor']['v0'],
                                          sensor_T_ISO_8855=label_dict['sensor']['sensor_T_ISO_8855'])

                K_matrix = np.zeros((4, 4))
                K_matrix[0][0] = label_dict['sensor']['fx']
                K_matrix[0][2] = label_dict['sensor']['u0']
                K_matrix[1][1] = label_dict['sensor']['fy']
                K_matrix[1][2] = label_dict['sensor']['v0']
                K_matrix[2][2] = 1

                label = {}
                label['calib'] = K_matrix
                label['orig'] = np.array(orig_img_size).astype(np.float32)

                label['clf'] = np.ones((max_objects, 1)) * 8
                label['c_3d'] = np.zeros((max_objects, 3))
                label['extent'] = np.zeros((max_objects, 3))
                label['bbox_3d'] = np.zeros((max_objects, 8, 3))
                label['bbox_2d'] = np.zeros((max_objects, 4))
                label['c_2d'] = np.zeros((max_objects, 2))
                label['ri'] = np.zeros((max_objects, 1))
                label['rj'] = np.zeros((max_objects, 1))
                for idx, obj in enumerate(label_dict['objects'][:max_objects]):
                    bbox = box_utils.Box3dImageTransform(camera)
                    bbox.initialize_box(center=obj['3d']['center'],
                                        quaternion=obj['3d']['rotation'],
                                        size=obj['3d']['dimensions'])

                    _, center_3d_cam, quaternion = bbox.get_parameters(coordinate_system=box_utils.CRS_S)

                    label['clf'][idx, 0] = CLASS_MAP[obj['label']]
                    label['c_3d'][idx, :] = center_3d_cam
                    label['extent'][idx, :] = [obj['3d']['dimensions'][2],  # height
                                               obj['3d']['dimensions'][1],  # width
                                               obj['3d']['dimensions'][0]]  # length

                    vertices = bbox.get_vertices(coordinate_system=box_utils.CRS_S)

                    for idx_vertice, loc in enumerate(bbox.loc):
                        label['bbox_3d'][idx, idx_vertice, :] = vertices[loc]
                    """
                        print(np.concatenate([vertices[loc], [1]], axis=0))
    
                        center = np.matmul(K_matrix, np.concatenate([vertices[loc], [1]]))
                        center = center[:2] / center[2]
    
                        img_arr[int(center[1]), int(center[0]), :] = (255, 255, 255)
                    cv.imshow("View0", img_arr)
                    cv.waitKey(0)
                    exit()"""

                    bbox_2d_xy_hw = obj['2d']['amodal']

                    bbox_2d_xy_xy = [bbox_2d_xy_hw[0],  # x_1
                                     bbox_2d_xy_hw[1],  # y_1
                                     bbox_2d_xy_hw[0]+bbox_2d_xy_hw[2],  # x_1 + width
                                     bbox_2d_xy_hw[1]+bbox_2d_xy_hw[3]]  # y_1 + height

                    label['c_2d'][idx, :] = [(bbox_2d_xy_xy[0]+((bbox_2d_xy_xy[2]-bbox_2d_xy_xy[0])/2.))/orig_img_size[1],
                                             (bbox_2d_xy_xy[1]+((bbox_2d_xy_xy[3]-bbox_2d_xy_xy[1])/2.))/orig_img_size[0]]

                    bbox_2d_xy_xy = np.array([np.clip(bbox_2d_xy_xy[0]/orig_img_size[1], 0, 1),
                                              np.clip(bbox_2d_xy_xy[1]/orig_img_size[0], 0, 1),
                                              np.clip(bbox_2d_xy_xy[2]/orig_img_size[1], 0, 1),
                                              np.clip(bbox_2d_xy_xy[3]/orig_img_size[0], 0, 1)])

                    label['bbox_2d'][idx, :] = bbox_2d_xy_xy

                    yaw, pitch, roll = pyquaternion.Quaternion(quaternion).yaw_pitch_roll
                    label['ri'][idx, 0] = np.cos(pitch)
                    label['rj'][idx, 0] = np.sin(pitch)

                scan = np.ones((5, 3))  # dummy points

                label = kitti_utils.remove_dontcare(label)

                tf_example = create_example(img, scan, label)

                writer.write(tf_example.SerializeToString())



if __name__ == '__main__':
    cfg = {
        'in_dir': '/cityscapes',
        'datasets': ['train', 'val'],
        'datasets_out': ['/tfrecords/cityscapes_train.tfrecord', '/tfrecords/cityscapes_val.tfrecord'],
        'n_scenes': -1,
        'img_size': (1024, 2048),
        'max_objects': 22,
    }

    create_records()
