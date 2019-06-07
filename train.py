import os
import sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR) # model
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import argparse
from datetime import datetime
import numpy as np
import tensorflow as tf
import importlib
import dataset
from io_util import *
import config

CONFIG = config.Config()

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='model_rpointnet', help='Model name [default: model_rpointnet]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--train_module', default='SPN', help='The module to be trained [options: SPN or RPOINTNET]')
parser.add_argument('--num_point', type=int, default=CONFIG.NUM_POINT, help='Point Number in a Scene [default: 18000]')
parser.add_argument('--num_point_ins', type=int, default=CONFIG.NUM_POINT_INS, help='Point Number of an Instance [default: 512]')
parser.add_argument('--num_category', type=int, default=CONFIG.NUM_CATEGORY, help='Maximum Number of Categories [default: 19]')
parser.add_argument('--num_sample', type=int, default=CONFIG.NUM_SAMPLE, help='Number of Sampled Seed Points [default: 256]')
parser.add_argument('--max_epoch', type=int, default=800, help='Epoch to run [default: 800]')
parser.add_argument('--batch_size', type=int, default=CONFIG.BATCH_SIZE, help='Batch Size during training [default: 2]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=55000, help='Decay step for lr decay [default: 55000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--restore_model_path', default=None, help='Restore model path e.g. log/model.ckpt')
parser.add_argument('--restore_scope', default=None, help='Restore variable scope')
parser.add_argument('--KL_weight', type=float, default=1, help='Additional weight for KL Loss')
parser.add_argument('--is_augment', type=int, default=1, help='Whether to augment the training data')
FLAGS = parser.parse_args()

EPOCH_CNT = 0

CONFIG.BATCH_SIZE = FLAGS.batch_size
CONFIG.NUM_POINT = FLAGS.num_point
CONFIG.NUM_POINT_INS = FLAGS.num_point_ins
CONFIG.NUM_CATEGORY = FLAGS.num_category
CONFIG.NUM_SAMPLE = FLAGS.num_sample
CONFIG.TRAIN_MODULE = [FLAGS.train_module]
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
KL_WEIGHT = FLAGS.KL_weight
IS_AUGMENT = FLAGS.is_augment

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

if not os.path.exists(os.path.join(ROOT_DIR, 'data/cache')):
    os.makedirs(os.path.join(ROOT_DIR, 'data/cache'))
SRC_MESH_PATH = os.path.join(ROOT_DIR, 'data/scannet_preprocessed/mesh/scans')
SRC_LABEL_PATH = os.path.join(ROOT_DIR, 'data/scannet_preprocessed/label/scans')
TRAIN_LIST = os.path.join(ROOT_DIR, 'data/scannet/scannet_train.txt')
VAL_LIST = os.path.join(ROOT_DIR, 'data/scannet/scannet_val.txt')
TRAIN_CACHE = os.path.join(ROOT_DIR, 'data/cache/train_%d_%d.npz'%(CONFIG.NUM_POINT, CONFIG.NUM_POINT_INS))
VAL_CACHE = os.path.join(ROOT_DIR, 'data/cache/val_%d_%d.npz'%(CONFIG.NUM_POINT, CONFIG.NUM_POINT_INS))
TRAIN_DATASET = dataset.ScanNetDataset(SRC_MESH_PATH, SRC_LABEL_PATH, TRAIN_LIST, TRAIN_CACHE, npoint=CONFIG.NUM_POINT, npoint_ins=CONFIG.NUM_POINT_INS, is_augment=IS_AUGMENT)
VAL_DATASET = dataset.ScanNetDataset(SRC_MESH_PATH, SRC_LABEL_PATH, VAL_LIST, VAL_CACHE, npoint=CONFIG.NUM_POINT, npoint_ins=CONFIG.NUM_POINT_INS, is_augment=False, permute_points=False)

CONFIG.NUM_GROUP = np.maximum(TRAIN_DATASET.ngroup, VAL_DATASET.ngroup)
TRAIN_DATASET.ngroup = CONFIG.NUM_GROUP
VAL_DATASET.ngroup = CONFIG.NUM_GROUP

def get_loss_weight(batch):
    alpha = 1.0*KL_WEIGHT - tf.train.exponential_decay(
                    1.0*KL_WEIGHT,  # Base learning rate.
                    batch * CONFIG.BATCH_SIZE,  # Current index into the dataset.
                    DECAY_STEP,          # Decay step.
                    DECAY_RATE,          # Decay rate.
                    staircase=True)
    return alpha

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * CONFIG.BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate        

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch * CONFIG.BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pc_pl, color_pl, pc_ins_pl, group_label_pl, group_indicator_pl, seg_label_pl, bbox_ins_pl = MODEL.placeholder_inputs(CONFIG)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            smpw_pl = tf.placeholder(tf.float32, shape=(CONFIG.BATCH_SIZE, CONFIG.NUM_POINT))
            
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            alpha = get_loss_weight(batch)
            tf.summary.scalar('bn_decay', bn_decay)
            tf.summary.scalar('alpha', alpha)
            print toYellow("---------- Get model and loss------------")
            # Get model and loss 
            end_points = MODEL.rpointnet(pc_pl, color_pl, pc_ins_pl, group_label_pl, group_indicator_pl, seg_label_pl, bbox_ins_pl, CONFIG, is_training_pl, mode='training', bn_decay=bn_decay)
            loss, end_points = MODEL.get_loss(end_points, CONFIG, alpha, smpw_pl, mode='training')

            tf.summary.scalar('loss', loss)
            tf.summary.scalar('spn_class_loss', end_points['spn_class_loss'])
            tf.summary.scalar('recons_loss', end_points['recons_loss'])
            tf.summary.scalar('shift_loss', end_points['shift_loss'])
            tf.summary.scalar('sem_loss', end_points['sem_loss'])
            tf.summary.scalar('kl_loss', end_points['kl_loss'])
            if 'RPOINTNET' in CONFIG.TRAIN_MODULE:
                tf.summary.scalar('rpointnet_class_loss', end_points['rpointnet_class_loss'])
                tf.summary.scalar('rpointnet_bbox_loss', end_points['rpointnet_bbox_loss'])
                tf.summary.scalar('rpointnet_mask_loss', end_points['rpointnet_mask_loss'])

            print toYellow("----------- Get training operator--------------")
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
        
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init)
        if FLAGS.restore_model_path is not None:
            if FLAGS.restore_scope is not None:
                loadvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=FLAGS.restore_scope)
            else:
                loadvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            saver_restore = tf.train.Saver(var_list=loadvars)
            saver_restore.restore(sess, FLAGS.restore_model_path)
            print('RESTORE MODEL FROM ' + FLAGS.restore_model_path)

        ops = {'pc_pl': pc_pl,
               'color_pl': color_pl,
               'pc_ins_pl': pc_ins_pl,
               'group_label_pl': group_label_pl,
               'group_indicator_pl': group_indicator_pl,
               'seg_label_pl': seg_label_pl,
               'bbox_ins_pl': bbox_ins_pl,
               'smpw_pl': smpw_pl,
               'is_training_pl': is_training_pl,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'end_points': end_points}

        best_loss = 1e20
        sess.graph.finalize()
        for epoch in range(MAX_EPOCH):
            log_string(toYellow('************ EPOCH %03d ***********' % (epoch)))
            log_string(toBlue('Training Model: ' + FLAGS.model))
            log_string(toGreen('Saving in: ' + LOG_DIR))
            sys.stdout.flush()
             
            train_one_epoch(sess, ops, train_writer)
            epoch_loss = eval_one_epoch(sess, ops, test_writer)
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                save_path = saver.save(sess, os.path.join(LOG_DIR, "best_model_epoch_%03d.ckpt"%(epoch)))
                log_string(toCyan("Model saved in file: %s" % save_path))

            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string(toCyan("Model saved in file: %s" % save_path))

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

def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    # Shuffle train samples
    train_idxs = np.arange(0, len(TRAIN_DATASET))
    np.random.shuffle(train_idxs)
    num_batches = len(TRAIN_DATASET)/CONFIG.BATCH_SIZE
    
    log_string(toYellow(str(datetime.now())))

    loss_sum = 0
    recons_loss_sum = 0
    kl_loss_sum = 0
    shift_loss_sum = 0
    sem_loss_sum = 0
    spn_class_loss_sum = 0
    rpointnet_class_loss_sum = 0
    rpointnet_bbox_loss_sum = 0
    rpointnet_mask_loss_sum = 0
    for batch_idx in range(num_batches):
        start_idx = batch_idx * CONFIG.BATCH_SIZE
        end_idx = (batch_idx+1) * CONFIG.BATCH_SIZE
        batch_pc, batch_color, batch_pc_ins, batch_group_label, batch_group_indicator, batch_seg_label, batch_bbox_ins, batch_smpw = \
            get_batch(TRAIN_DATASET, train_idxs, start_idx, end_idx)
        
        feed_dict = {ops['pc_pl']: batch_pc,
                     ops['color_pl']: batch_color,
                     ops['pc_ins_pl']: batch_pc_ins,
                     ops['group_label_pl']: batch_group_label,
                     ops['group_indicator_pl']: batch_group_indicator,
                     ops['seg_label_pl']: batch_seg_label,
                     ops['bbox_ins_pl']: batch_bbox_ins,
                     ops['smpw_pl']: batch_smpw,
                     ops['is_training_pl']: is_training}
        if 'RPOINTNET' in CONFIG.TRAIN_MODULE:
            summary, step, _, loss_val, recons_loss_val, kl_loss_val, shift_loss_val, sem_loss_val, spn_class_loss_val, rpointnet_class_loss_val, rpointnet_bbox_loss_val, rpointnet_mask_loss_val = sess.run([ops['merged'], ops['step'],
                ops['train_op'], ops['loss'], ops['end_points']['recons_loss'], ops['end_points']['kl_loss'],
                ops['end_points']['shift_loss'], ops['end_points']['sem_loss'], ops['end_points']['spn_class_loss'], ops['end_points']['rpointnet_class_loss'],
                ops['end_points']['rpointnet_bbox_loss'], ops['end_points']['rpointnet_mask_loss']], feed_dict=feed_dict)
        elif 'SPN' in CONFIG.TRAIN_MODULE:
            summary, step, _, loss_val, recons_loss_val, kl_loss_val, shift_loss_val, sem_loss_val, spn_class_loss_val = sess.run([ops['merged'], ops['step'],
                ops['train_op'], ops['loss'], ops['end_points']['recons_loss'], ops['end_points']['kl_loss'],
                ops['end_points']['shift_loss'], ops['end_points']['sem_loss'], ops['end_points']['spn_class_loss']], feed_dict=feed_dict)

        train_writer.add_summary(summary, step)
        loss_sum += loss_val
        recons_loss_sum += recons_loss_val
        kl_loss_sum += kl_loss_val
        shift_loss_sum += shift_loss_val
        sem_loss_sum += sem_loss_val
        spn_class_loss_sum += spn_class_loss_val
        if 'RPOINTNET' in CONFIG.TRAIN_MODULE:
            rpointnet_class_loss_sum += rpointnet_class_loss_val
            rpointnet_bbox_loss_sum += rpointnet_bbox_loss_val
            rpointnet_mask_loss_sum += rpointnet_mask_loss_val

        if (batch_idx+1)%10 == 0:
            log_string(toBlue(' -- %03d / %03d --' % (batch_idx+1, num_batches)))
            log_string(toYellow(' -- Model: ' + FLAGS.model))
            log_string(toGreen(' -- LOG DIR: ' + FLAGS.log_dir))
            log_string(toMagenta('mean loss: %f' % (loss_sum / 10)))
            log_string(toMagenta('mean reconstruction loss: %f' % (recons_loss_sum / 10)))
            log_string(toMagenta('mean kl-divergence loss: %f' % (kl_loss_sum / 10)))
            log_string(toMagenta('mean shift loss: %f' % (shift_loss_sum / 10)))
            log_string(toMagenta('mean sem loss: %f' % (sem_loss_sum / 10)))
            log_string(toMagenta('mean spn class loss: %f' % (spn_class_loss_sum / 10)))
            if 'RPOINTNET' in CONFIG.TRAIN_MODULE:
                log_string(toMagenta('mean rpointnet class loss: %f' % (rpointnet_class_loss_sum / 10)))
                log_string(toMagenta('mean rpointnet bbox loss: %f' % (rpointnet_bbox_loss_sum / 10)))
                log_string(toMagenta('mean rpointnet mask loss: %f' % (rpointnet_mask_loss_sum / 10)))
            loss_sum = 0
            recons_loss_sum = 0
            kl_loss_sum = 0
            shift_loss_sum = 0   
            sem_loss_sum = 0   
            spn_class_loss_sum = 0  
            rpointnet_class_loss_sum = 0
            rpointnet_bbox_loss_sum = 0
            rpointnet_mask_loss_sum = 0


def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    is_training = False
    test_idxs = np.arange(0, len(VAL_DATASET))
    num_batches = len(VAL_DATASET)/CONFIG.BATCH_SIZE

    log_string(toYellow(str(datetime.now())))
    log_string(toYellow('---- EPOCH %03d EVALUATION ----'%(EPOCH_CNT)))
    
    loss_sum = 0
    recons_loss_sum = 0
    kl_loss_sum = 0
    shift_loss_sum = 0
    sem_loss_sum = 0
    spn_class_loss_sum = 0
    rpointnet_class_loss_sum = 0
    rpointnet_bbox_loss_sum = 0
    rpointnet_mask_loss_sum = 0
    cum_intersection = np.zeros(18)
    cum_union = np.zeros(18)
    for batch_idx in range(num_batches):
        start_idx = batch_idx * CONFIG.BATCH_SIZE
        end_idx = (batch_idx+1) * CONFIG.BATCH_SIZE
        batch_pc, batch_color, batch_pc_ins, batch_group_label, batch_group_indicator, batch_seg_label, batch_bbox_ins, batch_smpw = get_batch(VAL_DATASET, test_idxs, start_idx, end_idx)

        feed_dict = {ops['pc_pl']: batch_pc,
                     ops['color_pl']: batch_color,
                     ops['pc_ins_pl']: batch_pc_ins,
                     ops['group_label_pl']: batch_group_label,
                     ops['group_indicator_pl']: batch_group_indicator,
                     ops['seg_label_pl']: batch_seg_label,
                     ops['bbox_ins_pl']: batch_bbox_ins,
                     ops['smpw_pl']: batch_smpw,
                     ops['is_training_pl']: is_training}
        if 'RPOINTNET' in CONFIG.TRAIN_MODULE:
            summary, step, loss_val, pred_val, sem_labels_val, recons_loss_val, kl_loss_val, shift_loss_val, sem_loss_val, spn_class_loss_val, rpointnet_class_loss_val, rpointnet_bbox_loss_val, rpointnet_mask_loss_val = sess.run([ops['merged'], ops['step'],
                ops['loss'], ops['end_points']['sem_class_logits'], ops['end_points']['sem_labels'], ops['end_points']['recons_loss'], ops['end_points']['kl_loss'], ops['end_points']['shift_loss'], ops['end_points']['sem_loss'], ops['end_points']['spn_class_loss'], ops['end_points']['rpointnet_class_loss'], ops['end_points']['rpointnet_bbox_loss'], ops['end_points']['rpointnet_mask_loss']], feed_dict=feed_dict)
        elif 'SPN' in CONFIG.TRAIN_MODULE:
            summary, step, loss_val, pred_val, sem_labels_val, recons_loss_val, kl_loss_val, shift_loss_val, sem_loss_val, spn_class_loss_val = sess.run([ops['merged'], ops['step'],
                ops['loss'], ops['end_points']['sem_class_logits'], ops['end_points']['sem_labels'], ops['end_points']['recons_loss'], ops['end_points']['kl_loss'], ops['end_points']['shift_loss'], ops['end_points']['sem_loss'], ops['end_points']['spn_class_loss']], feed_dict=feed_dict)
        pred_val = np.argmax(pred_val, 2) # BxN
        for s in range(18):
            cum_intersection[s] += np.sum(np.logical_and(pred_val==(s+1), sem_labels_val==(s+1)))
            cum_union[s] += np.sum(np.logical_or(pred_val==(s+1), sem_labels_val==(s+1)))

        test_writer.add_summary(summary, step)
        if 'RPOINTNET' in CONFIG.TRAIN_MODULE:
            loss_sum += (rpointnet_class_loss_val+rpointnet_bbox_loss_val+rpointnet_mask_loss_val)
        else:
            loss_sum += (recons_loss_val+kl_loss_val+shift_loss_val+spn_class_loss_val+sem_loss_val)
        recons_loss_sum += recons_loss_val
        kl_loss_sum += kl_loss_val
        shift_loss_sum += shift_loss_val
        sem_loss_sum += sem_loss_val
        spn_class_loss_sum += spn_class_loss_val
        if 'RPOINTNET' in CONFIG.TRAIN_MODULE:
            rpointnet_class_loss_sum += rpointnet_class_loss_val
            rpointnet_bbox_loss_sum += rpointnet_bbox_loss_val
            rpointnet_mask_loss_sum += rpointnet_mask_loss_val

    iou = np.divide(cum_intersection, cum_union+1e-8)
    meaniou = np.mean(iou)
    log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
    log_string('eval mean reconstruction loss: %f' % (recons_loss_sum / float(num_batches)))
    log_string('eval mean kl-divergence loss: %f' % (kl_loss_sum / float(num_batches)))
    log_string('eval mean shift loss: %f' % (shift_loss_sum / float(num_batches)))
    log_string('eval mean sem loss: %f' % (sem_loss_sum / float(num_batches)))
    log_string('eval mean iou: %f' % (meaniou))
    log_string('eval mean spn class loss: %f' % (spn_class_loss_sum / float(num_batches)))
    if 'RPOINTNET' in CONFIG.TRAIN_MODULE:
        log_string('eval mean rpointnet class loss: %f' % (rpointnet_class_loss_sum / float(num_batches)))
        log_string('eval mean rpointnet bbox loss: %f' % (rpointnet_bbox_loss_sum / float(num_batches)))
        log_string('eval mean rpointnet mask loss: %f' % (rpointnet_mask_loss_sum / float(num_batches)))
         
    EPOCH_CNT += 1
    if 'RPOINTNET' in CONFIG.TRAIN_MODULE:
        return (rpointnet_class_loss_sum + rpointnet_bbox_loss_sum + rpointnet_mask_loss_sum)/float(num_batches)
    elif 'SPN' in CONFIG.TRAIN_MODULE:
        return (recons_loss_sum)/float(num_batches)


if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()
