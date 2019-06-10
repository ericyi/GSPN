import os
import sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR) # model
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'external/ScanNet/BenchmarkScripts'))
import tensorflow as tf
import numpy as np
import argparse
import importlib
import dataset
import config
import io_util
import util_3d
from sklearn.neighbors import NearestNeighbors
import random
random.seed(0)

CONFIG = config.Config(istrain = False)

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--num_point', type=int, default=CONFIG.NUM_POINT, help='Point Number in a Scene [default: 18000]')
parser.add_argument('--num_point_ins', type=int, default=CONFIG.NUM_POINT_INS, help='Point Number of an Instance [default: 512]')
parser.add_argument('--num_category', type=int, default=CONFIG.NUM_CATEGORY, help='Maximum Number of Categories [default: 19]')
parser.add_argument('--num_sample', type=int, default=CONFIG.NUM_SAMPLE, help='Number of Sampled Seed Points [default: 2048]')
parser.add_argument('--model', default='model_rpointnet', help='Model name [default: model]')
parser.add_argument('--model_path', default='log/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
FLAGS = parser.parse_args()

CONFIG.NUM_POINT = FLAGS.num_point
CONFIG.NUM_POINT_INS = FLAGS.num_point_ins
CONFIG.NUM_CATEGORY = FLAGS.num_category
CONFIG.NUM_SAMPLE = FLAGS.num_sample
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
MODEL = importlib.import_module(FLAGS.model) # import network module

SRC_MESH_PATH = os.path.join(ROOT_DIR, 'data/scannet_preprocessed/mesh/scans')
SRC_LABEL_PATH = os.path.join(ROOT_DIR, 'data/scannet_preprocessed/label/scans')
VAL_LIST = os.path.join(ROOT_DIR, 'data/scannet/scannet_val.txt')
VAL_CACHE = os.path.join(ROOT_DIR, 'data/cache/val_%d_%d.npz'%(CONFIG.NUM_POINT, CONFIG.NUM_POINT_INS))
VAL_DATASET = dataset.ScanNetDataset(SRC_MESH_PATH, SRC_LABEL_PATH, VAL_LIST, VAL_CACHE, npoint=CONFIG.NUM_POINT, npoint_ins=CONFIG.NUM_POINT_INS, is_augment=False, permute_points=False)
CONFIG.NUM_GROUP = VAL_DATASET.ngroup

def get_model(batch_size):
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            CONFIG.BATCH_SIZE = batch_size
            pc_pl, color_pl, pc_ins_pl, group_label_pl, group_indicator_pl, seg_label_pl, bbox_ins_pl = MODEL.placeholder_inputs(CONFIG)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            smpw_pl = tf.placeholder(tf.float32, shape=(CONFIG.BATCH_SIZE, CONFIG.NUM_POINT))
            end_points = MODEL.rpointnet(pc_pl, color_pl, pc_ins_pl, group_label_pl, group_indicator_pl, seg_label_pl, bbox_ins_pl, CONFIG, is_training_pl, mode='inference', bn_decay=None)
            saver = tf.train.Saver()
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)
        # Restore variables from disk.
        saver.restore(sess, MODEL_PATH)
        ops = {'pc_pl': pc_pl,
               'color_pl': color_pl,
               'pc_ins_pl': pc_ins_pl,
               'group_label_pl': group_label_pl,
               'group_indicator_pl': group_indicator_pl,
               'seg_label_pl': seg_label_pl,
               'bbox_ins_pl': bbox_ins_pl,
               'smpw_pl': smpw_pl,
               'is_training_pl': is_training_pl,
               'end_points': end_points}
        return sess, ops

def get_batch(dataset, idxs, start_idx, end_idx):
    bsize = end_idx-start_idx
    batch_pc = np.zeros((bsize, CONFIG.NUM_POINT, 3))
    batch_color = np.zeros((bsize, CONFIG.NUM_POINT, 3))
    batch_pc_ins = np.zeros((bsize, CONFIG.NUM_GROUP, CONFIG.NUM_POINT_INS, 3))
    batch_group_label = np.zeros((bsize, CONFIG.NUM_POINT), dtype=np.int32)
    batch_group_indicator = np.zeros((bsize, CONFIG.NUM_GROUP), dtype=np.int32)
    batch_seg_label = np.zeros((bsize, CONFIG.NUM_POINT), dtype=np.int32)
    batch_bbox_ins = np.zeros((bsize, CONFIG.NUM_GROUP, 6), dtype=np.float32)
    batch_smpw = np.ones((bsize, CONFIG.NUM_POINT), dtype=np.float32)
    for i in range(bsize):
        pc, color, pc_ins, group_label, group_indicator, seg_label, bbox_ins = dataset[idxs[i+start_idx]]
        batch_pc[i,...] = pc
        batch_color[i,...] = color
        batch_pc_ins[i,...] = pc_ins
        batch_group_label[i,...] = group_label
        batch_group_indicator[i,...] = group_indicator
        batch_seg_label[i,...] = seg_label
        batch_bbox_ins[i,...] = bbox_ins
    return batch_pc, batch_color, batch_pc_ins, batch_group_label, batch_group_indicator, batch_seg_label, batch_bbox_ins, batch_smpw

def export_instance_ids_for_eval_customized(filename, label_ids, instance_ids, confidence):
    # label_ids: [NUM_INSTANCE]
    # instance_ids: [NUM_INSTANCE, NUM_POINT]
    # confidence: [NUM_INSTANCE]
    assert label_ids.shape[0] == instance_ids.shape[0]
    output_mask_path_relative = 'pred_mask'
    name = os.path.splitext(os.path.basename(filename))[0]
    output_mask_path = os.path.join(os.path.dirname(filename), output_mask_path_relative)
    if not os.path.isdir(output_mask_path):
        os.mkdir(output_mask_path)
    with open(filename, 'w') as f:
        for idx in np.arange(instance_ids.shape[0]):
            if np.sum(instance_ids[idx])==0: # 0 -> no instance for this vertex
                continue
            output_mask_file_relative = os.path.join(output_mask_path_relative, name + '_' + str(idx) + '.txt')
            f.write('%s %d %f\n' % (output_mask_file_relative, label_ids[idx], confidence[idx]))
            # write mask
            output_mask_file = os.path.join(output_mask_path, name + '_' + str(idx) + '.txt')
            util_3d.export_ids(output_mask_file, instance_ids[idx])

def gen_gt(dataset, scan_list_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    test_idxs = np.arange(0, len(dataset))
    scan_list = io_util.read_txt(scan_list_path)
    seg_label_map = np.array([0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
    for i, scan in enumerate(scan_list):
        print(i/float(len(scan_list)))
        batch_pc, batch_color, batch_pc_ins, batch_group_label, batch_group_indicator, batch_seg_label, batch_bbox_ins, batch_smpw = get_batch(dataset, test_idxs, i, i+1)
        batch_pc = np.squeeze(batch_pc, 0)
        batch_group_label = np.squeeze(batch_group_label, 0)
        batch_seg_label = np.squeeze(batch_seg_label, 0)
        batch_seg_label = seg_label_map[batch_seg_label]
        util_3d.export_ids(os.path.join(output_path, scan+'.txt'),batch_seg_label*1000+batch_group_label)

def output_prediction(sess, ops, dataset, scan_list_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    test_idxs = np.arange(0, len(dataset))
    scan_list = io_util.read_txt(scan_list_path)
    seg_label_map = np.array([0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
    sem_conf_th = 0.01
    fb_conf_th = 0.01
    mask_th = 0.4
    for i, scan in enumerate(scan_list):
        print(i/float(len(scan_list)))
        batch_pc, batch_color, batch_pc_ins, batch_group_label, batch_group_indicator, batch_seg_label, batch_bbox_ins, batch_smpw = get_batch(dataset, test_idxs, i, i+1)
        feed_dict = {ops['pc_pl']: batch_pc,
                     ops['color_pl']: batch_color,
                     ops['pc_ins_pl']: batch_pc_ins,
                     ops['group_label_pl']: batch_group_label,
                     ops['group_indicator_pl']: batch_group_indicator,
                     ops['seg_label_pl']: batch_seg_label,
                     ops['bbox_ins_pl']: batch_bbox_ins,
                     ops['smpw_pl']: batch_smpw,
                     ops['is_training_pl']: False}
        rpointnet_mask_selected, pc_coord_cropped_final_unnormalized, detections, sem_class_logits, pc_seed, fb_prob = sess.run([ops['end_points']['rpointnet_mask_selected'], 
            ops['end_points']['pc_coord_cropped_final_unnormalized'], ops['end_points']['detections'],
            ops['end_points']['sem_class_logits'], ops['end_points']['pc_seed'], ops['end_points']['fb_prob']], feed_dict=feed_dict)
        batch_pc = np.squeeze(batch_pc, 0) # [NUM_POINT, 3]
        rpointnet_mask_selected = np.squeeze(rpointnet_mask_selected, 0) # [DETECTION_MAX_INSTANCES, NUM_POINT_PER_ROI]
        pc_coord_cropped_final_unnormalized = np.squeeze(pc_coord_cropped_final_unnormalized, 0) # [DETECTION_MAX_INSTANCES, NUM_POINT_PER_ROI, 3]
        detections = np.squeeze(detections, 0) # [DETECTION_MAX_INSTANCES, 6+2]
        rpointnet_mask_selected = rpointnet_mask_selected[detections[:,6]>0,:]
        pc_coord_cropped_final_unnormalized = pc_coord_cropped_final_unnormalized[detections[:,6]>0,:,:]
        detections = detections[detections[:,6]>0,:]

        group_label_pred = np.zeros((detections.shape[0], batch_pc.shape[0]))
        for j in range(detections.shape[0]):
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(pc_coord_cropped_final_unnormalized[j,:,:])
            _, sidx = nbrs.kneighbors(batch_pc)
            sidx = np.reshape(sidx, -1)
            group_label_pred[j,:] = rpointnet_mask_selected[j,sidx]
            roi_mask = np.logical_and(batch_pc>=np.reshape(detections[j,:3]-detections[j,3:6]/2, [1,3]),
                batch_pc<=np.reshape(detections[j,:3]+detections[j,3:6]/2, [1,3]))
            roi_mask = np.logical_and(np.logical_and(roi_mask[:,0], roi_mask[:,1]), roi_mask[:,2])
            roi_mask = roi_mask.astype(np.float32)
            group_label_pred[j,:] = np.multiply(group_label_pred[j,:], roi_mask)
        #### get sem conf
        sem_class_logits = np.squeeze(sem_class_logits) # [N, NC]
        sem_class_prob = np.exp(sem_class_logits)
        sem_class_prob = np.divide(sem_class_prob, np.sum(sem_class_prob, -1, keepdims=True)) # [N, NC]
        sem_conf = np.matmul(group_label_pred, sem_class_prob) # [ND, NC]
        sem_conf = np.divide(sem_conf, 1e-8+np.sum(group_label_pred, 1, keepdims=True)) # [ND, NC]
        sem_conf = sem_conf.reshape(-1)[detections[:,6].astype(np.int32)+np.arange(detections.shape[0])*sem_conf.shape[1]] # [ND]
        #### get fb conf
        pc_seed = np.squeeze(pc_seed) # [NS, 3]
        fb_prob = np.squeeze(fb_prob) # [NS, 2]
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(pc_seed)
        _, sidx = nbrs.kneighbors(batch_pc)
        sidx = np.reshape(sidx, -1)
        fb_prob = fb_prob[sidx,1] # [N]
        fb_conf = np.matmul(group_label_pred, fb_prob) # [ND]
        fb_conf = np.divide(fb_conf, 1e-8+np.sum(group_label_pred, 1)) # [ND]

        #### Sort
        sidx = np.where(np.logical_and(sem_conf>sem_conf_th, fb_conf>fb_conf_th))[0]
        sidx2 = np.where(np.logical_or(sem_conf<=sem_conf_th, fb_conf<=fb_conf_th))[0]
        sidx_idx = np.argsort(detections[sidx,7]*sem_conf[sidx]*fb_conf[sidx])[::-1]
        sidx = sidx[sidx_idx]
        sidx2_idx = np.argsort(detections[sidx2,7]*sem_conf[sidx2]*fb_conf[sidx2])[::-1]
        sidx2 = sidx2[sidx2_idx]
        group_label_pred = np.concatenate((group_label_pred[sidx,:],group_label_pred[sidx2,:]),0)
        detections = np.concatenate((detections[sidx,:],detections[sidx2,:]),0)
        sem_conf = np.concatenate((sem_conf[sidx],sem_conf[sidx2]),0)
        fb_conf = np.concatenate((fb_conf[sidx],fb_conf[sidx2]),0)
        confidence_pred = np.concatenate((np.ones(len(sidx)), 0.97*np.ones(len(sidx2))),0)

        group_label_pred = (group_label_pred>mask_th).astype(np.int32)
        seg_label_pred = seg_label_map[(detections[:,6]).astype(np.int32)] #[NUM_INSTANCE]
        export_instance_ids_for_eval_customized(os.path.join(output_path, scan+'.txt'), seg_label_pred, group_label_pred, confidence_pred)


if __name__=='__main__':
    OUT_PATH = os.path.join(ROOT_DIR, 'eva')
    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)
    BATCH_SIZE = 1
    ##### Generate ground truth
    gen_gt(VAL_DATASET, VAL_LIST, os.path.join(OUT_PATH, 'gt'))
    
    ##### Generate prediction
    sess, ops = get_model(batch_size=1)
    output_prediction(sess, ops, VAL_DATASET, VAL_LIST, os.path.join(OUT_PATH, 'pred'))

    ##### Evaluate command
    # python evaluate_semantic_instance.py --pred_path root_dir/eva/pred --gt_path root_dir/eva/gt --output_file root_dir/eva/pred/semantic_instance_evaluation.txt
