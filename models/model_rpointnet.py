import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/nn_distance'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/sampling'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/grouping'))
import tensorflow as tf
import numpy as np
import tf_util
import tf_nndistance
from tf_sampling import farthest_point_sample, gather_point
from tf_grouping import query_ball_point, group_point
from pointnet_util import pointnet_fp_module, pointnet_sa_module
import config

def placeholder_inputs(config):
    pc_pl = tf.placeholder(tf.float32, shape=(config.BATCH_SIZE, config.NUM_POINT, 3))
    color_pl = tf.placeholder(tf.float32, shape=(config.BATCH_SIZE, config.NUM_POINT, 3))
    pc_ins_pl = tf.placeholder(tf.float32, shape=(config.BATCH_SIZE, config.NUM_GROUP, config.NUM_POINT_INS, 3))
    group_label_pl = tf.placeholder(tf.int32, shape=(config.BATCH_SIZE, config.NUM_POINT))
    group_indicator_pl = tf.placeholder(tf.int32, shape=(config.BATCH_SIZE, config.NUM_GROUP))
    seg_label_pl = tf.placeholder(tf.int32, shape=(config.BATCH_SIZE, config.NUM_POINT))
    bbox_ins_pl = tf.placeholder(tf.float32, shape=(config.BATCH_SIZE, config.NUM_GROUP, 6))
    return pc_pl, color_pl, pc_ins_pl, group_label_pl, group_indicator_pl, seg_label_pl, bbox_ins_pl

def multi_encoding_net(xyz, points, npoint, radius_list, nsample_list, mlp_list, mlp_list2, is_training, bn_decay, scope, bn=True, use_xyz=False, output_shift=False, shift_pred=None, fps_idx=None):
    ''' Encode multiple context.
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: list of float32 -- search radius in local region
            nsample: list of int32 -- how many points in each local region
            mlp: list of list of int32 -- output size for MLP on each point
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp2[-1]) TF tensor
            shift_pred: (batch_size, npoint, 3) TF tensor
            fps_idx: (batch_size, npoint) TF tensor
    '''
    with tf.variable_scope(scope) as sc:
        if fps_idx is None:
            fps_idx = farthest_point_sample(npoint, xyz) # (batch_size, npoint)
        new_xyz = gather_point(xyz, fps_idx) # (batch_size, npoint, 3)
        new_points_list = []
        group_xyz_list = []
        for i in range(len(radius_list)):
            radius = radius_list[i]
            nsample = nsample_list[i]
            idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)
            grouped_xyz = group_point(xyz, idx)
            grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1]) # [B, nseed, nsmp, 3]
            if shift_pred is not None:
                grouped_xyz -= tf.tile(tf.expand_dims(shift_pred, 2), [1,1,nsample,1])
            if points is not None:
                grouped_points = group_point(points, idx)
                if use_xyz:
                    grouped_points = tf.concat([grouped_points, grouped_xyz], axis=-1)
            else:
                grouped_points = grouped_xyz
            for j,num_out_channel in enumerate(mlp_list[i]):
                grouped_points = tf_util.conv2d(grouped_points, num_out_channel, [1, 1],
                                                padding='VALID', stride=[1,1], bn=bn, is_training=is_training,
                                                scope='conv_prev_%d_%d'%(i,j), bn_decay=bn_decay)
            new_points_list.append(tf.reduce_max(grouped_points, axis=[2]))
        new_points = tf.concat(new_points_list, axis=-1) # (batch_size, npoint, \sum_k{mlp[k][-1]})
        for i,num_out_channel in enumerate(mlp_list2):
            new_points = tf_util.conv1d(new_points, num_out_channel, 1,
                                               padding='VALID', stride=1, bn=bn, is_training=is_training,
                                               scope='conv_post_%d'%i, bn_decay=bn_decay)
        if output_shift:
            shift_pred = tf_util.conv1d(new_points, 4, 1,
                                        padding='VALID', stride=1, scope='conv_shift_pred', activation_fn=None)
        return new_xyz, new_points, shift_pred, fps_idx

def shift_pred_net(xyz, points, npoint_seed, end_points, scope, is_training, bn_decay=None, return_fullfea=False):
    ''' Encode multiple context.
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
        Return:
            pc_seed: (batch_size, npoint_seed, 3) TF tensor
            shift_pred_seed_4d: (batch_size, npoint_seed, 4) TF tensor
            ind_seed: (batch_size, npoint_seed) TF tensor
    '''
    with tf.variable_scope(scope) as sc:
        ind_seed = farthest_point_sample(npoint_seed, xyz) # (batch_size, npoint_seed)
        pc_seed = gather_point(xyz, ind_seed) # (batch_size, npoint_seed, 3)
        batch_size = xyz.get_shape()[0].value
        num_point = xyz.get_shape()[1].value
        l0_xyz = xyz
        l0_points = None # do not use color for shift prediction

        if return_fullfea:
            new_xyz = tf.concat((pc_seed, xyz), 1)
        else:
            new_xyz = pc_seed

        # Layer 1
        l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=2048, radius=0.2, nsample=32, mlp=[32,32,64], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1')
        l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=512, radius=0.4, nsample=32, mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2')
        l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=128, radius=0.8, nsample=32, mlp=[128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer3')
        l4_xyz, l4_points, l4_indices = pointnet_sa_module(l3_xyz, l3_points, npoint=32, radius=1.6, nsample=32, mlp=[256,256,512], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer4')

        # Feature Propagation layers
        l3_points = pointnet_fp_module(l3_xyz, l4_xyz, l3_points, l4_points, [256,256], is_training, bn_decay, scope='fa_layer1')
        l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [256,256], is_training, bn_decay, scope='fa_layer2')
        l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256,128], is_training, bn_decay, scope='fa_layer3')
        l0_points = pointnet_fp_module(new_xyz, l1_xyz, None, l1_points, [128,128,128], is_training, bn_decay, scope='fa_layer4')

        # FC layers
        net = tf_util.conv1d(l0_points, 4, 1,
                                    padding='VALID', stride=1, scope='conv_shift_pred', activation_fn=None)
        if return_fullfea:
            shift_pred_seed_4d, shift_pred_full_4d = tf.split(net, [npoint_seed, num_point], axis=1)
            end_points['shift_pred_full_4d'] = shift_pred_full_4d
        else:
            shift_pred_seed_4d = net

        end_points['pc_seed'] = pc_seed
        end_points['shift_pred_seed_4d'] = shift_pred_seed_4d
        end_points['ind_seed'] = ind_seed

        return end_points

def sem_net(xyz, points, npoint_sem, num_category, ind_seed, end_points, scope, is_training, bn_decay=None, return_fullfea=False):
    ''' Encode multiple context.
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint_sem: int32 -- #points to sample for fast training
            num_category: int32 -- #output category
            ind_seed: (batch_size, npoint_seed) sampling index of seed points
        Return:
            sem_fea_seed: (batch_size, npoint_seed, nfea)
            sem_fea: (batch_size, npoint_sem, nfea)
            sem_fea_full: (batch_size, ndataset, nfea)
            sem_class_logits: (batch_size, npoint_sem, num_category)
            ind_sem: (batch_size, npoint_sem) sampling index of sem points
    '''
    with tf.variable_scope(scope) as sc:
        batch_size = xyz.get_shape()[0].value
        num_point = xyz.get_shape()[1].value
        npoint_seed = ind_seed.get_shape()[1].value

        ind_sem = farthest_point_sample(npoint_sem, xyz) # (batch_size, npoint_sem)
        new_xyz_sem = gather_point(xyz, ind_sem) # (batch_size, npoint_sem, 3)
        new_points_sem = gather_point(points, ind_sem)
        end_points['ind_sem'] = ind_sem

        new_xyz_seed = gather_point(xyz, ind_seed) # (batch_size, npoint_seed, 3)
        new_points_seed = gather_point(points, ind_seed)

        if return_fullfea:
            new_xyz = tf.concat((new_xyz_seed, new_xyz_sem, xyz), 1)
            new_points = tf.concat((new_points_seed, new_points_sem, points), 1)
        else:
            new_xyz = tf.concat((new_xyz_seed, new_xyz_sem), 1)
            new_points = tf.concat((new_points_seed, new_points_sem), 1)
        
        l0_xyz = xyz
        l0_points = points

        # Layer 1
        l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=2048, radius=0.2, nsample=32, mlp=[32,32,64], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1')
        l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=512, radius=0.4, nsample=32, mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2')
        l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=128, radius=0.8, nsample=32, mlp=[128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer3')
        l4_xyz, l4_points, l4_indices = pointnet_sa_module(l3_xyz, l3_points, npoint=32, radius=1.6, nsample=32, mlp=[256,256,512], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer4')

        # with FPN
        if return_fullfea:
            end_points['sem_fea_full_l4'] = pointnet_fp_module(xyz, l4_xyz, points, l4_points, [], is_training, bn_decay, scope='fa_layer1_fpn')
            end_points['sem_fea_full_l3'] = pointnet_fp_module(xyz, l3_xyz, points, l3_points, [], is_training, bn_decay, scope='fa_layer2_fpn')
            end_points['sem_fea_full_l2'] = pointnet_fp_module(xyz, l2_xyz, points, l2_points, [], is_training, bn_decay, scope='fa_layer3_fpn')
            end_points['sem_fea_full_l1'] = pointnet_fp_module(xyz, l1_xyz, points, l1_points, [], is_training, bn_decay, scope='fa_layer4_fpn')

        # Feature Propagation layers
        l3_points = pointnet_fp_module(l3_xyz, l4_xyz, l3_points, l4_points, [256,256], is_training, bn_decay, scope='fa_layer1')
        l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [256,256], is_training, bn_decay, scope='fa_layer2')
        l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256,128], is_training, bn_decay, scope='fa_layer3')
        l0_points = pointnet_fp_module(new_xyz, l1_xyz, new_points, l1_points, [128,128,128], is_training, bn_decay, scope='fa_layer4')

        # FC layers
        net = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
        if return_fullfea:
            sem_fea_seed, sem_fea, sem_fea_full = tf.split(net, [npoint_seed, npoint_sem, num_point], axis=1)
            end_points['sem_fea_seed'] = sem_fea_seed
            end_points['sem_fea'] = sem_fea
            end_points['sem_fea_full'] = sem_fea_full
        else:
            sem_fea_seed, sem_fea = tf.split(net, [npoint_seed, npoint_sem], axis=1)
            end_points['sem_fea_seed'] = sem_fea_seed
            end_points['sem_fea'] = sem_fea

        # net = end_points['sem_fea_full']
        net = end_points['sem_fea']
        net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
        sem_class_logits = tf_util.conv1d(net, num_category, 1, padding='VALID', activation_fn=None, scope='fc2')
        end_points['sem_class_logits'] = sem_class_logits

        return end_points


def pn2_fea_extractor(xyz, points, scope, is_training, bn_decay=None):
    ''' Encode multiple context.
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
        Return:
            new_points: (batch_size, ndataset, channel_out) TF tensor
    '''
    with tf.variable_scope(scope) as sc:
        batch_size = xyz.get_shape()[0].value
        num_point = xyz.get_shape()[1].value
        l0_xyz = xyz
        l0_points = points

        # Layer 1
        l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=2048, radius=0.2, nsample=32, mlp=[32,32,64], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1')
        l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=512, radius=0.4, nsample=32, mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2')
        l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=128, radius=0.8, nsample=32, mlp=[128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer3')

        # Feature Propagation layers
        l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [256,128], is_training, bn_decay, scope='fa_layer1')
        l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [128,64], is_training, bn_decay, scope='fa_layer2')
        new_points = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, l1_points, [64,64,64], is_training, bn_decay, scope='fa_layer3')

        return new_points


def single_encoding_net(pc, mlp_list, mlp_list2, scope, is_training, bn_decay):
    ''' The encoding network for instance
    Input:
        pc: [B, N, 3]
    Return:
        net: [B, nfea]
    '''
    with tf.variable_scope(scope) as myscope:
        net = tf.expand_dims(pc, 2)
        for i,num_out_channel in enumerate(mlp_list):
            net = tf_util.conv2d(net, num_out_channel, [1,1],
                                 padding='VALID', stride=[1,1],
                                 bn=True, is_training=is_training,
                                 scope='conv%d'%i, bn_decay=bn_decay)
        net = tf.reduce_max(net, axis=[1])
        net = tf.squeeze(net, 1)
        for i,num_out_channel in enumerate(mlp_list2):
            net = tf_util.fully_connected(net, num_out_channel, bn=True, is_training=is_training,
                                          scope='fc%d'%i, bn_decay=bn_decay)
        return net

def fea_trans_net(input_fea, mlp_list, scope, is_training, bn_decay):
    with tf.variable_scope(scope) as myscope:
        net = input_fea
        nlayer = len(mlp_list)
        for i,num_out_channel in enumerate(mlp_list):
            if i<nlayer-1:
                net = tf_util.conv1d(net, num_out_channel, 1, padding='VALID', bn=True, is_training=is_training,
                                     scope='conv%d'%i, bn_decay=bn_decay)
            else:
                net = tf_util.conv1d(net, num_out_channel, 1, padding='VALID', activation_fn=None, scope='conv%d'%i)
        return net

def sample(mean, log_var):
    # Sample z
    z = mean + tf.exp(log_var/2.0) * tf.random_normal(tf.shape(mean), 0, 1, dtype=tf.float32)
    return z

def decoding_net(feat, num_point, scope, is_training, bn_decay):
    ''' The decoding network for shape generation
    Input:
        feat: [B, nsmp, nfea]
    Return:
        pc: [B, nsmp, num_point, 3]
    '''
    with tf.variable_scope(scope) as myscope:
        nsmp = feat.get_shape()[1].value
        nfea = feat.get_shape()[2].value
        feat = tf.reshape(feat, [-1, nfea])
        # UPCONV Decoder
        if num_point<=3072 and num_point>1536:
            conv_feat = tf.expand_dims(tf.expand_dims(feat, 1),1)
            net = tf_util.conv2d_transpose(conv_feat, 512, kernel_size=[2,2], stride=[1,1], padding='VALID', scope='upconv1', bn=True, bn_decay=bn_decay, is_training=is_training)
            net = tf_util.conv2d_transpose(net, 256, kernel_size=[3,3], stride=[1,1], padding='VALID', scope='upconv2', bn=True, bn_decay=bn_decay, is_training=is_training)
            net = tf_util.conv2d_transpose(net, 256, kernel_size=[4,4], stride=[2,2], padding='VALID', scope='upconv3', bn=True, bn_decay=bn_decay, is_training=is_training)
            net = tf_util.conv2d_transpose(net, 128, kernel_size=[5,5], stride=[3,3], padding='VALID', scope='upconv4', bn=True, bn_decay=bn_decay, is_training=is_training)
            net = tf_util.conv2d_transpose(net, 3, kernel_size=[1,1], stride=[1,1], padding='VALID', scope='upconv5', activation_fn=None)
            num_point_conv = 1024
        elif num_point<=1536 and num_point>896:
            conv_feat = tf.expand_dims(tf.expand_dims(feat, 1),1)
            net = tf_util.conv2d_transpose(conv_feat, 512, kernel_size=[2,2], stride=[1,1], padding='VALID', scope='upconv1', bn=True, bn_decay=bn_decay, is_training=is_training)
            net = tf_util.conv2d_transpose(net, 256, kernel_size=[2,2], stride=[1,1], padding='VALID', scope='upconv2', bn=True, bn_decay=bn_decay, is_training=is_training)
            net = tf_util.conv2d_transpose(net, 256, kernel_size=[3,3], stride=[2,2], padding='VALID', scope='upconv3', bn=True, bn_decay=bn_decay, is_training=is_training)
            net = tf_util.conv2d_transpose(net, 128, kernel_size=[4,4], stride=[3,3], padding='VALID', scope='upconv4', bn=True, bn_decay=bn_decay, is_training=is_training)
            net = tf_util.conv2d_transpose(net, 3, kernel_size=[1,1], stride=[1,1], padding='VALID', scope='upconv5', activation_fn=None)
            num_point_conv = 484
        elif num_point<=896 and num_point>384:
            conv_feat = tf.expand_dims(tf.expand_dims(feat, 1),1)
            net = tf_util.conv2d_transpose(conv_feat, 512, kernel_size=[3,3], stride=[1,1], padding='VALID', scope='upconv1', bn=True, bn_decay=bn_decay, is_training=is_training)
            net = tf_util.conv2d_transpose(net, 256, kernel_size=[3,3], stride=[2,2], padding='VALID', scope='upconv2', bn=True, bn_decay=bn_decay, is_training=is_training)
            net = tf_util.conv2d_transpose(net, 128, kernel_size=[4,4], stride=[2,2], padding='VALID', scope='upconv3', bn=True, bn_decay=bn_decay, is_training=is_training)
            net = tf_util.conv2d_transpose(net, 3, kernel_size=[1,1], stride=[1,1], padding='VALID', scope='upconv4', activation_fn=None)
            num_point_conv = 256
        else:
            raise('Exception')
        pc_upconv = tf.reshape(net, [-1, num_point_conv, 3])

        num_point_fc = num_point - num_point_conv
        # FC Decoder
        net = tf_util.fully_connected(feat, 512, bn=True, is_training=is_training, scope='de_fc2', bn_decay=bn_decay)
        net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training, scope='de_fc3', bn_decay=bn_decay)
        net = tf_util.fully_connected(net, num_point_fc*3, activation_fn=None, scope='de_fc4')
        pc_fc = tf.reshape(net, [-1, num_point_fc, 3])
        # Merge
        pc = tf.concat([pc_upconv, pc_fc], axis=1)
        pc = tf.reshape(pc, [-1, nsmp, num_point, 3])
        return pc

def shape_proposal_net(pc, color, pc_ins, group_label, group_indicator, num_category, scope, is_training, bn_decay=None, nsmp=128, return_fullfea=False):
    ''' Shape proposal generation
    Inputs:
        pc: [B, NUM_POINT, 3]
        color: [B, NUM_POINT, 3]
        pc_ins: [B, NUM_GROUP, NUM_POINT_INS, 3], in world coord sys
        group_label: [B, NUM_POINT]
        group_indicator: [B, NUM_GROUP]
    Returns:
        fb_logits: [B, NUM_SAMPLE, 2] confidence logits (before softmax)
        fb_prob: [B, NUM_SAMPLE, 2] confidence probabilities
        bbox_ins: [B, NUM_SAMPLE, (x, y, z, l, w, h)]
        entity_fea: [B, NUM_POINT, nfea] entity feature for each point
        center_pos: [B, NUM_POINT, 3] center coordinate for each point, in world coord sys
    '''
    with tf.variable_scope(scope) as myscope:
        # Parameter extraction
        batch_size = pc.get_shape()[0].value
        ngroup = pc_ins.get_shape()[1].value
        nsmp_ins = pc_ins.get_shape()[2].value
        end_points = {}

        # Shift prediction, ind_seed [B, nsmp], shift_pred_seed [B, nsmp, 3]
        end_points = shift_pred_net(pc, color, nsmp, end_points, 'shift_predictor', is_training, bn_decay=bn_decay, return_fullfea=return_fullfea)
        pc_seed = end_points['pc_seed']
        shift_pred_seed_4d = end_points['shift_pred_seed_4d']
        ind_seed = end_points['ind_seed']
        shift_pred_seed = tf.multiply(shift_pred_seed_4d[:,:,:3], shift_pred_seed_4d[:,:,3:])

        # Semantic prediction, sem_fea_seed [B, nsmp, nfea]
        end_points = sem_net(pc, color, 1024, num_category, ind_seed, end_points, 'sem_predictor', is_training, bn_decay=bn_decay, return_fullfea=return_fullfea)
        sem_fea_seed = end_points['sem_fea_seed']

        # Encode instance, pcfea_ins_centered [B, ngroup, nfea_ins], pc_ins_center [B, ngroup, 1, 3]
        pc_ins_center = (tf.reduce_max(pc_ins, 2, keep_dims=True)+tf.reduce_min(pc_ins, 2, keep_dims=True))/2 # [B, ngroup, 1, 3] -> requires random padding for pc_ins generation
        pc_ins_centered = pc_ins-pc_ins_center
        idx = tf.where(tf.greater(group_indicator, 0))
        pc_ins_centered_list = tf.gather_nd(pc_ins_centered, idx)
        pcfea_ins_centered_list = single_encoding_net(pc_ins_centered_list, [64, 256, 512], [256], 'instance_encoder', is_training, bn_decay)
        nfea_ins = pcfea_ins_centered_list.get_shape()[1].value
        pcfea_ins_centered = tf.scatter_nd(tf.cast(idx,tf.int32), pcfea_ins_centered_list, tf.constant([batch_size, ngroup, nfea_ins])) # [B, ngroup, nfea_ins]
        
        # Collect instance feature for seed points [B, nsmp, nfea_seed]
        idx = tf.where(tf.greater_equal(ind_seed,0))
        ind_seed_aug = tf.concat((tf.expand_dims(tf.cast(idx[:,0],tf.int32),-1),tf.reshape(ind_seed,[-1,1])),1)
        group_label_seed = tf.reshape(tf.gather_nd(group_label, ind_seed_aug), [-1, nsmp]) # [B, nsmp]
        idx = tf.where(tf.greater_equal(group_label_seed,0))
        group_label_seed_aug = tf.concat((tf.expand_dims(tf.cast(idx[:,0],tf.int32),-1),tf.reshape(group_label_seed,[-1,1])),1)
        pcfea_ins_seed = tf.reshape(tf.gather_nd(pcfea_ins_centered, group_label_seed_aug), [-1, nsmp, nfea_ins])
        pc_ins_centered_seed = tf.reshape(tf.gather_nd(pc_ins_centered, group_label_seed_aug), [-1, nsmp, nsmp_ins, 3])
        pc_ins_center_seed = tf.reshape(tf.gather_nd(pc_ins_center, group_label_seed_aug), [-1, nsmp, 1, 3])
       
        # Encode context, pcfea_seed [B, nsmp, nfea_context]
        _, pcfea_seed, _, _ = multi_encoding_net(pc, color, nsmp, [0.5,1.0,1.5], [256,256,512], [[64,128,256], [64,128,256], [64,128,256]], [], is_training, bn_decay, scope='context_encoder', use_xyz=True, output_shift=False, shift_pred=tf.stop_gradient(shift_pred_seed), fps_idx=ind_seed)

        # Compute foreground/background score [B, nsmp, 2]
        fb_logits = fea_trans_net(pcfea_seed, [256, 64, 2], 'fb_logits', is_training, bn_decay)
        fb_prob = tf.nn.softmax(fb_logits, -1)

        # Compute mu and sigma [B, nsmp, 512]
        mu_sigma_c = fea_trans_net(tf.concat((sem_fea_seed, pcfea_seed), axis=-1), [256, 512, 512], 'mu_sigma_c', is_training, bn_decay)
        mu_sigma_x = fea_trans_net(tf.concat((sem_fea_seed, pcfea_seed, pcfea_ins_seed), axis=-1), [256, 512, 512], 'mu_sigma_x', is_training, bn_decay)
        
        # Sample z [B, nsmp, 256]
        mean = mu_sigma_x[:,:,:256]
        log_var = mu_sigma_x[:,:,256:]
        log_var = tf.clip_by_value(log_var, -10.0, 1.0)
        cmean = mu_sigma_c[:,:,:256]
        clog_var = mu_sigma_c[:,:,256:]
        clog_var = tf.clip_by_value(clog_var, -10.0, 1.0)
        zi = sample(mean, log_var)
        zc = cmean
        z = tf.cond(is_training, lambda: zi, lambda: zc)
        
        # Decode shapes pc [B, nsmp, nsmp_ins, 3]
        gcfeat = tf_util.conv1d(pcfea_seed, 256, 1, padding='VALID', bn=True, is_training=is_training,
                                scope='dec_fc', bn_decay=bn_decay)
        feat = tf.concat((z, gcfeat), axis=-1)
        pc_ins_pred = decoding_net(feat, nsmp_ins, 'decoder', is_training=is_training, bn_decay=bn_decay)
        pc_ins_pred = pc_ins_pred + tf.stop_gradient(tf.expand_dims(shift_pred_seed, 2))

        # Collect bbox for reconstructions
        pc_ins_pred_world_coord = pc_ins_pred + tf.expand_dims(pc_seed, 2)
        bbox_ins_pred = tf.concat(((tf.reduce_max(pc_ins_pred_world_coord, 2)+tf.reduce_min(pc_ins_pred_world_coord, 2))/2, 
                                  tf.reduce_max(pc_ins_pred_world_coord, 2)-tf.reduce_min(pc_ins_pred_world_coord, 2)), 2) # [B, nsmp, 6] -> center + l,w,h

        # Propagate seed feature and center position
        if return_fullfea:
            entity_fea = pointnet_fp_module(pc, pc_seed, None, pcfea_seed, [], is_training=False, bn_decay=None, scope='entity_fea_prop', bn=False)
            shift_pred = tf.multiply(end_points['shift_pred_full_4d'][:,:,:3], end_points['shift_pred_full_4d'][:,:,3:])
            center_pos = pc+shift_pred
            end_points['entity_fea'] = entity_fea # [B, N, 256] entity feature of each point
            end_points['center_pos'] = center_pos # [B, N, 3] center location in the world coord sys

        # Store end_points
        end_points['shift_pred_seed'] = shift_pred_seed # [B, nsmp, 3], offset from seed point to ins center
        end_points['shift_pred_seed_4d'] = shift_pred_seed_4d # [B, nsmp, 4], offset from seed point to ins center
        end_points['pc_seed'] = pc_seed # [B, nsmp, 3], seed point coordinate in world coord sys
        end_points['ind_seed'] = ind_seed # [B, nsmp], seed index
        end_points['pc_ins_centered_seed'] = pc_ins_centered_seed # [B, nsmp, nsmp_ins, 3], centered gt instance point cloud for each seed
        end_points['pc_ins_center_seed'] = pc_ins_center_seed # [B, nsmp, 1, 3], gt instance center for each seed in world coord sys
        end_points['mean'] = mean # [B, nsmp, 256]
        end_points['log_var'] = log_var
        end_points['cmean'] = cmean
        end_points['clog_var'] = clog_var
        end_points['fb_logits'] = fb_logits # [B, nsmp, 2] foreground/backgroud logits
        end_points['fb_prob'] = fb_prob # [B, nsmp, 2] foreground/background probability
        end_points['pc_ins_pred'] = pc_ins_pred # [B, nsmp, nsmp_ins, 3], in local sys, needs to add pc_seed
        end_points['bbox_ins_pred'] = bbox_ins_pred # [B, nsmp, 6]

        return end_points

def nms_3d(boxes, scores, pre_nms_limit, max_output_size, iou_threshold=0.5, score_threshold=float('-inf')):
    ''' Non maximum suppression in 3D
    Inputs:
        boxes: [B, N, 6] center + l,w,h
        scores: [B, N] prob between 0 and 1
    Outputs:
        selected_indices: [B, M]
    '''

    batch_size = scores.shape[0]    
    num_box_input = scores.shape[1]    
    sidx = np.argsort(-scores, 1) # [B, N] from large to small
    selected_indices = -np.ones((batch_size, max_output_size), dtype=np.int32) # [B, M]
    for i in range(batch_size):
        cursidx = sidx[i,:]
        curscores = scores[i,:]
        curvolume = boxes[i,:,3]*boxes[i,:,4]*boxes[i,:,5]
        if pre_nms_limit>0:
            cursidx = cursidx[:pre_nms_limit]
        cursidx = cursidx[curscores[cursidx]>score_threshold]
        count = 0
        while len(cursidx)>0 and count<max_output_size:
            selected_indices[i,count] = cursidx[0]
            count += 1
            vA = np.maximum(boxes[i,[cursidx[0]],:3]-boxes[i,[cursidx[0]],3:]/2, boxes[i,cursidx,:3]-boxes[i,cursidx,3:]/2)
            vB = np.minimum(boxes[i,[cursidx[0]],:3]+boxes[i,[cursidx[0]],3:]/2, boxes[i,cursidx,:3]+boxes[i,cursidx,3:]/2)
            intersection_cube = np.maximum(vB-vA,0)
            intersection_volume = intersection_cube[:,0]*intersection_cube[:,1]*intersection_cube[:,2]
            iou = np.divide(intersection_volume,curvolume[cursidx]+curvolume[cursidx[0]]-intersection_volume+1e-8)
            cursidx = np.delete(cursidx, np.where(iou>iou_threshold)[0])
    return selected_indices

def gather_selection(source, selected_idx, max_selected_size):
    '''
    Inputs:
        source: [B, N, C]
        selected_idx: [B, M], -1 means not selecting anything
    Returns:
        target: [B, M, C], 0 padded
    '''
    batch_size = source.get_shape()[0].value
    fea_size = source.get_shape()[2].value
    pos_idx = tf.cast(tf.where(tf.greater_equal(selected_idx,0)), tf.int32)
    selected_idx_vec = tf.gather_nd(selected_idx, pos_idx)
    target_vec = tf.gather_nd(source, tf.concat((tf.expand_dims(pos_idx[:,0],-1), tf.reshape(selected_idx_vec,[-1,1])),1))
    target = tf.scatter_nd(pos_idx, target_vec, tf.constant([batch_size, max_selected_size, fea_size]))
    return target

def trim_zeros_graph(boxes, name=None):
    """Often boxes are represented with matrices of shape [N, 4] and
    are padded with zeros. This removes zero boxes.

    boxes: [N, 6] matrix of boxes.
    non_zeros: [N] a 1D boolean mask identifying the rows to keep
    """
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros

def batch_slice(inputs, graph_fn, batch_size, names=None):
    """Splits inputs into slices and feeds each slice to a copy of the given
    computation graph and then combines the results. It allows you to run a
    graph on a batch of inputs even if the graph is written to support one
    instance only.
    inputs: list of tensors. All must have the same first dimension length
    graph_fn: A function that returns a TF tensor that's part of a graph.
    batch_size: number of slices to divide the data into.
    names: If provided, assigns names to the resulting tensors.
    """
    if not isinstance(inputs, list):
        inputs = [inputs]
    outputs = []
    for i in range(batch_size):
        inputs_slice = [x[i] for x in inputs]
        output_slice = graph_fn(*inputs_slice)
        if not isinstance(output_slice, (tuple, list)):
            output_slice = [output_slice]
        outputs.append(output_slice)
    # Change outputs from a list of slices where each is
    # a list of outputs to a list of outputs and each has
    # a list of slices
    outputs = list(zip(*outputs))

    if names is None:
        names = [None] * len(outputs)

    result = [tf.stack(o, axis=0, name=n)
              for o, n in zip(outputs, names)]
    if len(result) == 1:
        result = result[0]

    return result

def box_shrink(box, pc):
    ''' Shrink bounding box so that it is tight with respect to pc
    Inputs:
        box: [B, NUM_SAMPLE, 6]
        pc: [B, NUM_POINT, 3]
    Returns:
        box: [B, NUM_SAMPLE, 6]
    '''
    pc_aug = tf.expand_dims(pc, 1) # [B, 1, NUM_POINT, 3]
    box_aug = tf.expand_dims(box, 2) # [B, NUM_SAMPLE, 1, 6]
    box_masks = tf.logical_and(pc_aug>=(box_aug[:,:,:,:3]-box_aug[:,:,:,3:]/2),
                               pc_aug<=(box_aug[:,:,:,:3]+box_aug[:,:,:,3:]/2)) # [B, NUM_SAMPLE, NUM_POINT, 3]
    box_masks = tf.logical_and(tf.logical_and(box_masks[:,:,:,0], box_masks[:,:,:,1]), box_masks[:,:,:,2]) # [B, NUM_SAMPLE, NUM_POINT]
    box_out_masks = 1-tf.cast(tf.expand_dims(box_masks, -1), tf.float32) # [B, NUM_SAMPLE, NUM_POINT, 1]
    gamma = 1e4 # a large number for the box estimation trick
    box_max = tf.reduce_max(pc_aug-gamma*box_out_masks,2) # [B, NUM_SAMPLE, 3]
    box_min = tf.reduce_min(pc_aug+gamma*box_out_masks,2)
    box = tf.concat( ((box_max+box_min)/2, box_max-box_min+1e-3), 2) # [B, NUM_SAMPLE, 6]
    keep = tf.greater(box_max-box_min, 0)
    keep = tf.logical_and(tf.logical_and(keep[:,:,0], keep[:,:,1]), keep[:,:,2])
    keep = tf.expand_dims(tf.cast(keep, tf.float32), -1)
    box = tf.multiply(box, keep)
    return box

def box_refinement(box, gt_box):
    """Compute refinement needed to transform box to gt_box.
    box and gt_box are [N, (center_x, center_y, center_z, l, w, h)]
    """
    box = tf.cast(box, tf.float32)
    gt_box = tf.cast(gt_box, tf.float32)

    dz = (gt_box[:,2] - box[:,2]) / (box[:,5]+1e-8)
    dy = (gt_box[:,1] - box[:,1]) / (box[:,4]+1e-8)
    dx = (gt_box[:,0] - box[:,0]) / (box[:,3]+1e-8)
    dh = tf.log(gt_box[:,5] / (box[:,5]+1e-8))
    dw = tf.log(gt_box[:,4] / (box[:,4]+1e-8))
    dl = tf.log(gt_box[:,3] / (box[:,3]+1e-8))

    result = tf.stack([dz, dy, dx, dh, dw, dl], axis=1)
    return result

def apply_box_delta(box, delta):
    ''' Apply bounding box delta to refine box
    Inputs:
        box: [NUM_SAMPLE, 6]
        delta: [NUM_SAMPLE, 6]
    Returns:
        box_refined: [NUM_SAMPLE, 6]
    '''
    delta = tf.stack([delta[:,2], delta[:,1], delta[:,0], delta[:,5], delta[:,4], delta[:,3]], axis=1)
    box_refined_part2 = tf.multiply(tf.exp(delta[:,3:]), box[:,3:])
    box_refined_part1 = tf.multiply(delta[:,:3], box[:,3:])+box[:,:3]
    box_refined = tf.concat((box_refined_part1, box_refined_part2), axis=1)
    return box_refined

def sample_points_within_box(masks, nsmp):
    '''
    Inputs:
        masks: [nmask, npoint]
        nsmp: scalar
    Returns:
        masks_selection_idx: [nmask, nsmp]
    '''
    if masks.shape[0]==0:
        return np.zeros((0, nsmp), dtype=np.int32)
    else:
        masks_selection_idx = [np.random.choice(np.where(masks[i,:])[0], nsmp, replace=True) for i in np.arange(masks.shape[0])]
        masks_selection_idx = np.stack(masks_selection_idx, 0).astype(np.int32)
        return masks_selection_idx

def spn_target_gen(proposals, proposal_seed_class_ids, gt_class_ids, gt_boxes):
    ''' SPN target generation
    Inputs:
        proposals: [NUM_SAMPLE, 6]
        proposal_seed_class_ids: [NUM_SAMPLE] - 0 is background class and 1 is foreground class
        gt_class_ids: [NUM_GROUP] - 0 is background class and 1 is foreground class
        gt_boxes: [NUM_GROUP, 6]
    Returns:
        spn_match: [NUM_SAMPLE], 1 = positive, -1 = negative, 0 = neutral
    '''
    # Remove zero padding
    non_zeros = tf.where(tf.greater(gt_class_ids, 0))[:,0]
    gt_boxes = tf.gather(gt_boxes, non_zeros, axis=0, name="spn_trim_gt_boxes")
    gt_class_ids = tf.gather(gt_class_ids, non_zeros, axis=0, name="spn_trim_gt_class_ids")

    # Compute IoU [NUM_SAMPLE, NUM_GROUP_TRIMMED]
    proposals_aug = tf.expand_dims(proposals, 1)
    gt_boxes_aug = tf.expand_dims(gt_boxes, 0)
    proposal_volume = tf.multiply(tf.multiply(proposals_aug[:,:,3], proposals_aug[:,:,4]), proposals_aug[:,:,5])
    gt_boxes_volume = tf.multiply(tf.multiply(gt_boxes_aug[:,:,3], gt_boxes_aug[:,:,4]), gt_boxes_aug[:,:,5])
    vA = tf.maximum(proposals_aug[:,:,:3]-proposals_aug[:,:,3:]/2, gt_boxes_aug[:,:,:3]-gt_boxes_aug[:,:,3:]/2)
    vB = tf.minimum(proposals_aug[:,:,:3]+proposals_aug[:,:,3:]/2, gt_boxes_aug[:,:,:3]+gt_boxes_aug[:,:,3:]/2)
    intersection_cube = tf.maximum(vB-vA,0)
    intersection_volume = tf.multiply(tf.multiply(intersection_cube[:,:,0], intersection_cube[:,:,1]), intersection_cube[:,:,2])
    ious = tf.divide(intersection_volume,proposal_volume+gt_boxes_volume-intersection_volume+1e-8)

    # Determine positive and negative ROIs
    roi_iou_max = tf.reduce_max(ious, axis=1)
    # 1. Positive ROIs are those with >=0.5 IoU with a GT box and gt class id is 1
    # We also guarantee that the boxes with the largest IoU with a GT box is also treated as positive
    spn_match_positive = tf.logical_and(tf.greater_equal(roi_iou_max, 0.5), tf.equal(proposal_seed_class_ids,1))
    masked_ious = tf.multiply(ious, tf.cast(tf.equal(tf.expand_dims(proposal_seed_class_ids,-1),1), tf.float32))
    positive_aug_idx = tf.argmax(masked_ious,0)
    positive_aug_idx = tf.boolean_mask(positive_aug_idx, tf.reduce_max(masked_ious, 0)>0)
    updates = tf.cast(tf.ones_like(positive_aug_idx), tf.float32)
    spn_match_positive_aug = tf.cond(
        tf.size(positive_aug_idx)>0,
        true_fn = lambda: tf.scatter_nd(positive_aug_idx, updates, spn_match_positive.shape),
        false_fn = lambda: tf.constant(0.0))
    spn_match_positive = tf.greater(tf.cast(spn_match_positive, tf.float32)+spn_match_positive_aug, 0)
    # 2. Negative ROIs are those with < 0.5 with every GT box and not a postive ROI
    spn_match_negative = tf.logical_and(tf.less(roi_iou_max, 0.5), tf.logical_not(spn_match_positive))

    spn_match = tf.cast(spn_match_positive, tf.float32)-tf.cast(spn_match_negative, tf.float32)

    return spn_match


def detection_target_gen(proposals, gt_class_ids, gt_boxes, gt_masks, pc, config):
    ''' Generate detection targets for training
    Inputs:
        proposals: [SPN_NMS_MAX_SIZE, 6], zero padded
        gt_class_ids: [NUM_GROUP] - 0 is background class and objects start from 1
        gt_boxes: [NUM_GROUP, 6]
        gt_masks: [NUM_POINT, NUM_GROUP]
        pc: [NUM_POINT, 3]
    Returns:
        rois: [TRAIN_ROIS_PER_IMAGE, 6], zero padded
        roi_gt_class_ids: [TRAIN_ROIS_PER_IMAGE] - 0 is invalid class and used for padding, zero padded
        deltas: [TRAIN_ROIS_PER_IMAGE, 6], zero padded
        masks_selection_idx: [TRAIN_ROIS_PER_IMAGE, NUM_POINT_INS_MASK] - which points are selected from the initial point cloud, zero padded
        masks: [TRAIN_ROIS_PER_IMAGE, NUM_POINT_INS_MASK] - binary mask, zero padded
    '''
    # Remove zero padding
    proposals, _ = trim_zeros_graph(proposals, name='trim_proposals')
    gt_boxes, non_zeros = trim_zeros_graph(gt_boxes, name="trim_gt_boxes")
    gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros,
                                   name="trim_gt_class_ids")
    gt_masks = tf.gather(gt_masks, tf.where(non_zeros)[:, 0], axis=1,
                         name="trim_gt_masks")

    # Remove empty proposals
    pc_aug = tf.expand_dims(pc,1) # [N, 1, 3]
    proposals_aug = tf.expand_dims(proposals, 0) # [1, NP, 6]
    roi_masks = tf.logical_and(pc_aug>=(proposals_aug[:,:,:3]-proposals_aug[:,:,3:]/2),
                               pc_aug<=(proposals_aug[:,:,:3]+proposals_aug[:,:,3:]/2))
    roi_masks = tf.logical_and(tf.logical_and(roi_masks[:,:,0], roi_masks[:,:,1]), roi_masks[:,:,2]) # [N, NP]
    non_empty_idx = tf.where(tf.greater(tf.reduce_sum(tf.cast(roi_masks, tf.float32), 0), 0))[:,0]
    roi_masks = tf.gather(roi_masks, non_empty_idx, axis=1) # [N, NP']
    proposals = tf.gather(proposals, non_empty_idx, axis=0) # [NP', 6]

    # Compute IoU [n_proposal, n_gt_boxes]
    proposals_aug = tf.expand_dims(proposals, 1)
    gt_boxes_aug = tf.expand_dims(gt_boxes, 0)
    proposal_volume = tf.multiply(tf.multiply(proposals_aug[:,:,3], proposals_aug[:,:,4]), proposals_aug[:,:,5])
    gt_boxes_volume = tf.multiply(tf.multiply(gt_boxes_aug[:,:,3], gt_boxes_aug[:,:,4]), gt_boxes_aug[:,:,5])
    vA = tf.maximum(proposals_aug[:,:,:3]-proposals_aug[:,:,3:]/2, gt_boxes_aug[:,:,:3]-gt_boxes_aug[:,:,3:]/2)
    vB = tf.minimum(proposals_aug[:,:,:3]+proposals_aug[:,:,3:]/2, gt_boxes_aug[:,:,:3]+gt_boxes_aug[:,:,3:]/2)
    intersection_cube = tf.maximum(vB-vA,0)
    intersection_volume = tf.multiply(tf.multiply(intersection_cube[:,:,0], intersection_cube[:,:,1]), intersection_cube[:,:,2])
    ious = tf.divide(intersection_volume,proposal_volume+gt_boxes_volume-intersection_volume+1e-8)
    
    # Determine positive and negative ROIs
    roi_iou_max = tf.reduce_max(ious, axis=1)
    # 1. Positive ROIs are those with >= 0.5 IoU with a GT box
    positive_indices = tf.where(roi_iou_max >= 0.5)[:, 0]
    # 2. Negative ROIs are those with < 0.5 with every GT box
    negative_indices = tf.where(roi_iou_max < 0.5)[:, 0]

    # Subsample ROIs. Aim for 33% positive
    # Positive ROIs
    positive_count = int(config.TRAIN_ROIS_PER_IMAGE *
                         config.ROI_POSITIVE_RATIO)
    positive_indices = tf.random_shuffle(positive_indices)[:positive_count]
    positive_count = tf.shape(positive_indices)[0]
    # Negative ROIs. Add enough to maintain positive:negative ratio.
    r = 1.0 / config.ROI_POSITIVE_RATIO
    negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
    negative_indices = tf.random_shuffle(negative_indices)[:negative_count]
    # Gather selected ROIs [POSITIVE_COUNT/NEGATIVE_COUNT, 6]
    positive_rois = tf.gather(proposals, positive_indices)
    negative_rois = tf.gather(proposals, negative_indices)

    # Assign positive ROIs to GT boxes. roi_gt_boxes: [POSITIVE_COUNT, 6], roi_gt_class_ids: [POSITIVE_COUNT]
    positive_ious = tf.gather(ious, positive_indices)
    roi_gt_box_assignment = tf.cond(
        tf.greater(tf.shape(positive_ious)[1], 0),
        true_fn = lambda: tf.argmax(positive_ious, axis=1),
        false_fn = lambda: tf.cast(tf.constant([]),tf.int64)
    )
    roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)
    roi_gt_class_ids = tf.gather(gt_class_ids, roi_gt_box_assignment)

    # Compute bbox refinement for positive ROIs. delta: [POSITIVE_COUNT, 6]
    deltas = box_refinement(positive_rois, roi_gt_boxes)
    deltas /= config.BBOX_STD_DEV

    # Compute mask targets
    roi_gt_masks = tf.cast(tf.gather(gt_masks, roi_gt_box_assignment, axis=1), tf.bool) # [NUM_POINT, POSITIVE_COUNT]
    positive_roi_masks = tf.gather(roi_masks, positive_indices, axis=1)
    masks_full = tf.transpose(tf.logical_and(positive_roi_masks, roi_gt_masks)) # [POSITIVE_COUNT, NUM_POINT]
    masks_selection_idx = tf.stop_gradient(tf.py_func(sample_points_within_box, 
        [tf.transpose(positive_roi_masks), config.NUM_POINT_INS_MASK], tf.int32)) # [POSITIVE_COUNT, NUM_POINT_INS_MASK]
    smp_idx = tf.reshape(tf.tile(tf.reshape(tf.range(positive_count),[-1,1]),[1, config.NUM_POINT_INS_MASK]),[-1,1])
    smp_idx = tf.concat((smp_idx, tf.reshape(masks_selection_idx,[-1,1])),1)
    masks = tf.reshape(tf.gather_nd(masks_full, smp_idx),[-1, config.NUM_POINT_INS_MASK]) # [POSITIVE_COUNT, NUM_POINT_INS_MASK]

    # Append negative ROIs and pad bbox deltas and masks that
    # are not used for negative ROIs with zeros.
    rois = tf.concat([positive_rois, negative_rois], axis=0)
    N = tf.shape(negative_rois)[0]
    P = tf.maximum(config.TRAIN_ROIS_PER_IMAGE - tf.shape(rois)[0], 0)
    rois = tf.pad(rois, [(0, P), (0, 0)])
    roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, N + P)])
    deltas = tf.pad(deltas, [(0, N + P), (0, 0)])
    masks_selection_idx = tf.pad(masks_selection_idx, [[0, N + P], (0, 0)])
    masks = tf.pad(masks, [[0, N + P], (0, 0)])

    return rois, roi_gt_class_ids, deltas, masks_selection_idx, masks

def mask_selection_gen(proposals, pc, num_rois, config, empty_removal=True):
    ''' Generate detection targets for training
    Inputs:
        proposals: [SPN_NMS_MAX_SIZE, 6], zero padded
        pc: [NUM_POINT, 3]
    Returns:
        proposals: [NUM_ROIS, 6]
        masks_selection_idx: [NUM_ROIS, NUM_POINT_INS_MASK] - which points are selected from the initial point cloud, zero padded
    '''
    # Remove zero padding
    proposals, _ = trim_zeros_graph(proposals, name='trim_proposals')

    # Remove empty proposals
    pc_aug = tf.expand_dims(pc,1) # [N, 1, 3]
    proposals_aug = tf.expand_dims(proposals, 0) # [1, NP, 6]
    roi_masks = tf.logical_and(pc_aug>=(proposals_aug[:,:,:3]-proposals_aug[:,:,3:]/2-1e-3),
                               pc_aug<=(proposals_aug[:,:,:3]+proposals_aug[:,:,3:]/2+1e-3))
    roi_masks = tf.logical_and(tf.logical_and(roi_masks[:,:,0], roi_masks[:,:,1]), roi_masks[:,:,2]) # [N, NP]
    if empty_removal:
        non_empty_idx = tf.where(tf.greater(tf.reduce_sum(tf.cast(roi_masks, tf.float32), 0), 0))[:,0]
        roi_masks = tf.gather(roi_masks, non_empty_idx, axis=1) # [N, NP']
        proposals = tf.gather(proposals, non_empty_idx, axis=0) # [NP', 6]
    proposals_count = tf.shape(proposals)[0]

    # Generate mask selection index
    masks_selection_idx = tf.stop_gradient(tf.py_func(sample_points_within_box, 
        [tf.transpose(roi_masks), config.NUM_POINT_INS_MASK], tf.int32)) # [NP', NUM_POINT_INS_MASK]

    # Append negative ROIs and pad bbox deltas and masks that
    # are not used for negative ROIs with zeros.
    P = tf.maximum(num_rois - proposals_count, 0)
    masks_selection_idx = tf.pad(masks_selection_idx, [[0, P], (0, 0)])
    proposals = tf.pad(proposals, [(0, P), (0, 0)])

    return proposals, masks_selection_idx

def points_cropping(pc, pc_fea, pc_center, rois, masks_selection_idx, num_rois, num_point_per_roi, normalize_crop_region=True):
    ''' Crop points for network heads, in analogy to ROIAlign
    Inputs:
        pc: [B, NUM_POINT, 3]
        pc_fea: [B, NUM_POINT, NFEA]
        pc_center: [B, NUM_POINT, 3]
        rois: [B, NUM_ROIS, 6], zero padded
        masks_selection_idx: [B, NUM_ROIS, NUM_POINT_PER_ROI]
    Returns:
        pc_fea_cropped: [B, NUM_ROIS, NUM_POINT_PER_ROI, NFEA]
        pc_center_cropped: [B, NUM_ROIS, NUM_POINT_PER_ROI, 3]
        pc_coord_cropped: [B, NUM_ROIS, NUM_POINT_PER_ROI, 3]
    '''
    batch_size = pc.get_shape()[0].value
    smp_idx = tf.reshape(tf.tile(tf.reshape(tf.range(batch_size),[-1,1]),[1, num_rois*num_point_per_roi]),[-1,1])
    smp_idx = tf.concat((smp_idx, tf.reshape(masks_selection_idx,[-1,1])),1)
    pc_fea_cropped = tf.reshape(tf.gather_nd(pc_fea, smp_idx),[batch_size, num_rois, num_point_per_roi, -1])
    pc_center_cropped = tf.reshape(tf.gather_nd(pc_center, smp_idx),[batch_size, num_rois, num_point_per_roi, -1])
    pc_coord_cropped_unnormalized = tf.reshape(tf.gather_nd(pc, smp_idx),[batch_size, num_rois, num_point_per_roi, -1])
    pc_coord_cropped = pc_coord_cropped_unnormalized

    # convert world coord to local
    rois_center = tf.expand_dims(rois[:,:,:3], 2)
    pc_coord_cropped = pc_coord_cropped-rois_center
    pc_center_cropped = pc_center_cropped-rois_center
    if normalize_crop_region:
        # scale box to [1,1,1]
        rois = rois+tf.cast(tf.equal(tf.reduce_sum(rois, 2, keep_dims=True),0),tf.float32)
        rois_size = tf.expand_dims(rois[:,:,3:], 2)
        pc_coord_cropped = tf.divide(pc_coord_cropped, rois_size)
        pc_center_cropped = tf.divide(pc_center_cropped, rois_size)
    return pc_fea_cropped, pc_center_cropped, pc_coord_cropped, pc_coord_cropped_unnormalized

def refine_detections(rois, probs, deltas, pc, fb_prob, sem_prob, config):
    '''Refine classified proposals and filter overlaps and return final
    detections.
    Inputs:
        rois: [NUM_ROIS, 6], zero padded, in world coord sys
        probs: [NUM_ROIS, NUM_CATEGORY] - 0 is background class and objects start from 1
        deltas: [NUM_ROIS, NUM_CATEGORY, 6]
        pc: [NUM_POINT, 3]
        fb_prob: [NUM_ROIS]
        sem_prob: [NUM_ROIS]
    Returns:
        detections: [NUM_DETECTIONS, (center_x, center_y, center_z, l, w, h, class_id, score)]
    '''
    # Class IDs per ROI
    class_ids = tf.argmax(probs, axis=1, output_type=tf.int32)
    # Class probability of the top class of each ROI
    indices = tf.stack([tf.range(probs.shape[0]), class_ids], axis=1)
    class_scores = tf.gather_nd(probs, indices) # [NUM_ROIS]
    # Class-specific bounding box deltas
    deltas_specific = tf.gather_nd(deltas, indices) # [NUM_ROIS, 6]
    # Apply bounding box deltas

    # Shape: [NUM_ROIS, (center_x, center_y, center_z, l, w, h)]
    refined_rois = apply_box_delta(rois, deltas_specific * config.BBOX_STD_DEV)
    # Shrink boxes
    if config.SHRINK_BOX:
        refined_rois = tf.squeeze(box_shrink(tf.expand_dims(refined_rois, 0), tf.expand_dims(pc,0)),0)

    # Filter out background boxes
    keep = tf.where(class_ids > 0)[:, 0]
    # Filter out low confidence boxes
    if config.DETECTION_MIN_CONFIDENCE:
        conf_keep = tf.where(class_scores >= config.DETECTION_MIN_CONFIDENCE)[:, 0]
        keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                                        tf.expand_dims(conf_keep, 0))
        keep = tf.sparse_tensor_to_dense(keep)[0]

    # Apply per-class NMS
    # 1. Prepare variables
    pre_nms_class_ids = tf.gather(class_ids, keep)
    # pre_nms_scores = tf.gather(class_scores, keep) # [NUM_KEEP]
    pre_nms_scores = tf.gather(class_scores*fb_prob*sem_prob, keep) # [NUM_KEEP]
    pre_nms_rois = tf.gather(refined_rois,   keep)
    unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

    def nms_keep_map(class_id):
        """Apply Non-Maximum Suppression on ROIs of the given class."""
        # Indices of ROIs of the given class
        ixs = tf.where(tf.equal(pre_nms_class_ids, class_id))[:, 0]
        # Apply NMS

        class_keep = tf.py_func(nms_3d, [tf.expand_dims(tf.gather(pre_nms_rois, ixs),0),
            tf.expand_dims(tf.gather(pre_nms_scores, ixs),0),
            -1, config.DETECTION_MAX_INSTANCES,
            config.DETECTION_NMS_THRESHOLD, float('-inf')], tf.int32)
        class_keep = tf.squeeze(class_keep, 0) # [DETECTION_MAX_INSTANCES], -1 padded
        class_keep = tf.gather(class_keep, tf.where(class_keep > -1)[:,0]) # [<=DETECTION_MAX_INSTANCES], no padding

        # Map indices
        class_keep = tf.gather(keep, tf.gather(ixs, class_keep))
        # Pad with -1 so returned tensors have the same shape
        gap = config.DETECTION_MAX_INSTANCES - tf.shape(class_keep)[0]
        class_keep = tf.pad(class_keep, [(0, gap)],
                            mode='CONSTANT', constant_values=-1)
        # Set shape so map_fn() can infer result shape
        class_keep.set_shape([config.DETECTION_MAX_INSTANCES])
        return class_keep

    # 2. Map over class IDs
    nms_keep = tf.map_fn(nms_keep_map, unique_pre_nms_class_ids,
                         dtype=tf.int64)
    # 3. Merge results into one list, and remove -1 padding
    nms_keep = tf.reshape(nms_keep, [-1])
    nms_keep = tf.gather(nms_keep, tf.where(nms_keep > -1)[:, 0])
    # 4. Compute intersection between keep and nms_keep
    keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                                    tf.expand_dims(nms_keep, 0))
    keep = tf.sparse_tensor_to_dense(keep)[0]
    # Keep top detections
    roi_count = config.DETECTION_MAX_INSTANCES
    class_scores_keep = tf.gather(class_scores*fb_prob*sem_prob, keep)
    num_keep = tf.minimum(tf.shape(class_scores_keep)[0], roi_count)
    top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]
    keep = tf.gather(keep, top_ids)

    # Arrange output as [N, (center_x, center_y, center_z, l, w, h, class_id, score)]
    detections = tf.concat([
        tf.gather(refined_rois, keep),
        tf.to_float(tf.gather(class_ids, keep))[..., tf.newaxis],
        tf.gather(class_scores, keep)[..., tf.newaxis]
        ], axis=1)

    # Pad with zeros if detections < DETECTION_MAX_INSTANCES
    gap = config.DETECTION_MAX_INSTANCES - tf.shape(detections)[0]
    detections = tf.pad(detections, [(0, gap), (0, 0)], "CONSTANT")
    return detections

def classification_head(pc, pc_fea, num_category, mlp_list, mlp_list2, is_training, bn_decay, scope, bn=True):
    ''' Classification head for both class id prediction and bbox delta regression
    Inputs:
        pc: [B, NUM_ROIS, NUM_POINT_PER_ROI, 3]
        pc_fea: [B, NUM_ROIS, NUM_POINT_PER_ROI, NFEA]
        num_category: scalar
    Returns:
        logits: [B, NUM_ROIS, NUM_CATEGORY]
        probs: [B, NUM_ROIS, NUM_CATEGORY]
        bbox_deltas: [B, NUM_ROIS, NUM_CATEGORY, (dz, dy, dx, log(dh), log(dw), log(dl))]
    '''
    with tf.variable_scope(scope) as myscope:
        num_rois = pc.get_shape()[1].value
        grouped_points = tf.concat((pc_fea, pc), -1)
        for i,num_out_channel in enumerate(mlp_list):
            grouped_points = tf_util.conv2d(grouped_points, num_out_channel, [1, 1],
                                            padding='VALID', stride=[1,1], bn=bn, is_training=is_training,
                                            scope='conv_prev_%d'%i, bn_decay=bn_decay)
        new_points = tf.reduce_max(grouped_points, axis=2)
        for i,num_out_channel in enumerate(mlp_list2):
            new_points = tf_util.conv1d(new_points, num_out_channel, 1,
                                        padding='VALID', stride=1, bn=bn, is_training=is_training,
                                        scope='conv_post_%d'%i, bn_decay=bn_decay)
        logits = tf_util.conv1d(new_points, num_category, 1, padding='VALID',
                                stride=1, scope='conv_classify', activation_fn=None)
        probs = tf.nn.softmax(logits, 2)
        bbox_deltas = tf_util.conv1d(new_points, num_category*6, 1, padding='VALID',
                                     stride=1, scope='conv_bbox_regress', activation_fn=None)
        bbox_deltas = tf.reshape(bbox_deltas, [-1, num_rois, num_category, 6])
        return logits, probs, bbox_deltas

def segmentation_head(pc, pc_fea, num_category, mlp_list, mlp_list2, mlp_list3, is_training, bn_decay, scope, bn=True):
    ''' Segmentation head
    Inputs:
        pc: [B, NUM_ROIS, NUM_POINT_PER_ROI, 3]
        pc_fea: [B, NUM_ROIS, NUM_POINT_PER_ROI, NFEA]
        num_category: scalar
    Returns:
        masks: [B, NUM_ROIS, NUM_POINT_PER_ROI, NUM_CATEGORY]
    '''
    with tf.variable_scope(scope) as myscope:
        num_rois = pc.get_shape()[1].value
        num_point_per_roi = pc.get_shape()[2].value
        grouped_points = tf.concat((pc_fea, pc), -1)
        for i,num_out_channel in enumerate(mlp_list):
            grouped_points = tf_util.conv2d(grouped_points, num_out_channel, [1, 1],
                                            padding='VALID', stride=[1,1], bn=bn, is_training=is_training,
                                            scope='conv_prev_%d'%i, bn_decay=bn_decay)
        local_feat = grouped_points
        for i,num_out_channel in enumerate(mlp_list2):
            grouped_points = tf_util.conv2d(grouped_points, num_out_channel, [1, 1],
                                            padding='VALID', stride=[1,1], bn=bn, is_training=is_training,
                                            scope='conv_%d'%i, bn_decay=bn_decay)
        global_feat = tf.reduce_max(grouped_points, axis=2, keep_dims=True)
        global_feat_expanded = tf.tile(global_feat, [1, 1, num_point_per_roi, 1])
        new_points = tf.concat((global_feat_expanded, local_feat), -1)
        for i,num_out_channel in enumerate(mlp_list3):
            new_points = tf_util.conv2d(new_points, num_out_channel, [1, 1],
                                            padding='VALID', stride=[1,1], bn=bn, is_training=is_training,
                                            scope='conv_post_%d'%i, bn_decay=bn_decay)

        masks = tf_util.conv2d(new_points, num_category, [1, 1], padding='VALID',
                               stride=[1,1], scope='conv_seg', activation_fn=None)
        return masks

def dict_stop_gradient(dict_in):
    keys = dict_in.keys()
    for key in keys:
        dict_in[key] = tf.stop_gradient(dict_in[key])
    return dict_in

def select_segmentation(rpointnet_masks, class_ids):
    ''' Convert segmentation into point cloud label
    Inputs:
        rpointnet_masks: [B, NUM_ROIS, NUM_POINT_PER_ROI, NUM_CATEGORY]
        class_ids: [B, NUM_ROIS]
    Returns:
        rpointnet_mask_selected: [B, NUM_ROIS, NUM_POINT_PER_ROI]
    '''
    batch_size = rpointnet_masks.get_shape()[0].value
    num_rois = rpointnet_masks.get_shape()[1].value
    num_point_per_roi = rpointnet_masks.get_shape()[2].value
    num_category = rpointnet_masks.get_shape()[3].value

    rpointnet_masks = tf.reshape(rpointnet_masks, [-1, num_point_per_roi, num_category])
    rpointnet_masks = tf.transpose(rpointnet_masks, perm=[0, 2, 1]) # [B*NUM_ROIS, NUM_CATEGORY, NUM_POINT_PER_ROI]
    class_ids = tf.cast(tf.reshape(class_ids, [-1]), tf.int32)
    class_ids_aug = tf.stack([tf.range(batch_size*num_rois, dtype=tf.int32), class_ids], 1)
    rpointnet_mask_selected = tf.gather_nd(rpointnet_masks, class_ids_aug) #[-1, NUM_POINT_PER_ROI]
    rpointnet_mask_selected = tf.reshape(rpointnet_mask_selected, [batch_size, num_rois, num_point_per_roi])

    return rpointnet_mask_selected

def unmold_segmentation(rpointnet_masks, rois, class_ids, pc_coord_cropped, pc):
    ''' Convert segmentation into point cloud label
    Inputs:
        rpointnet_masks: [B, NUM_ROIS, NUM_POINT_PER_ROI, NUM_CATEGORY]
        rois: [B, NUM_ROIS, 6]
        class_ids: [B, NUM_ROIS]
        pc_coord_cropped: [B, NUM_ROIS, NUM_POINT_PER_ROI, 3]
        pc: [B, NUM_POINT, 3]
    Returns:
        rpointnet_mask_unmolded: [B, NUM_ROIS, NUM_POINT]
    '''
    batch_size = rpointnet_masks.get_shape()[0].value
    num_rois = rpointnet_masks.get_shape()[1].value
    num_point_per_roi = rpointnet_masks.get_shape()[2].value
    num_category = rpointnet_masks.get_shape()[3].value
    num_point = pc.get_shape()[1].value

    rpointnet_masks = tf.reshape(rpointnet_masks, [-1, num_point_per_roi, num_category])
    rpointnet_masks = tf.transpose(rpointnet_masks, perm=[0, 2, 1]) # [B*NUM_ROIS, NUM_CATEGORY, NUM_POINT_PER_ROI]
    class_ids = tf.cast(tf.reshape(class_ids, [-1]), tf.int32)
    class_ids_aug = tf.stack([tf.range(batch_size*num_rois, dtype=tf.int32), class_ids], 1)
    rpointnet_masks = tf.gather_nd(rpointnet_masks, class_ids_aug) #[-1, NUM_POINT_PER_ROI]

    # [B, NUM_ROIS, NUM_POINT, NUM_POINT_PER_ROI]
    dist = tf.reduce_sum(tf.square(tf.expand_dims(tf.expand_dims(pc, 1),3)-tf.expand_dims(pc_coord_cropped,2)), -1)
    min_idx = tf.argmin(dist, 3, output_type=tf.int32) # [B, NUM_ROIS, NUM_POINT]
    min_idx = tf.reshape(min_idx, [-1, num_point]) # [-1, NUM_POINT]
    min_idx_aug = tf.tile(tf.expand_dims(tf.range(batch_size*num_rois, dtype=tf.int32),-1), [1, num_point])
    min_idx_aug = tf.stack([tf.reshape(min_idx_aug, [-1]), tf.reshape(min_idx, [-1])], 1)
    rpointnet_mask_unmolded = tf.reshape(tf.gather_nd(rpointnet_masks, min_idx_aug), [batch_size, num_rois, num_point])

    # Mask out regions outside rois
    pc_aug = tf.expand_dims(pc, 1) # [B, 1, NUM_POINT, 3]
    rois_aug = tf.expand_dims(rois, 2) # [B, NUM_ROIS, 1, 6]
    roi_masks = tf.logical_and(pc_aug>=(rois_aug[:,:,:,:3]-rois_aug[:,:,:,3:]/2),
                               pc_aug<=(rois_aug[:,:,:,:3]+rois_aug[:,:,:,3:]/2))
    roi_masks = tf.logical_and(tf.logical_and(roi_masks[:,:,:,0], roi_masks[:,:,:,1]), roi_masks[:,:,:,2]) # [B, NUM_ROIS, NUM_POINT]
    roi_masks = tf.cast(roi_masks, tf.float32)

    rpointnet_mask_unmolded = tf.multiply(rpointnet_mask_unmolded, roi_masks)
    return rpointnet_mask_unmolded


def rpointnet(pc, color, pc_ins, group_label, group_indicator, seg_label, bbox_ins, config, is_training, mode='training', bn_decay=None):
    ''' Shape proposal generation
    Inputs:
        pc: [B, NUM_POINT, 3]
        color: [B, NUM_POINT, 3]
        pc_ins: [B, NUM_GROUP, NUM_POINT_INS, 3], in world coord sys
        group_label: [B, NUM_POINT]
        group_indicator: [B, NUM_GROUP]
        seg_label: [B, NUM_POINT]
        bbox_ins: [B, NUM_GROUP, 6]
    Returns:
        
    '''
    assert mode in ['training', 'inference']
    if not config.USE_COLOR:
        color = None
    if 'SPN' in config.TRAIN_MODULE and mode=='training':
        end_points = shape_proposal_net(pc, color, pc_ins, group_label, group_indicator, config.NUM_CATEGORY, scope='shape_proposal_net', is_training=is_training, bn_decay=bn_decay, nsmp=config.NUM_SAMPLE, return_fullfea=False)
    else:
        end_points = shape_proposal_net(pc, color, pc_ins, group_label, group_indicator, config.NUM_CATEGORY, scope='shape_proposal_net', is_training=tf.constant(False), bn_decay=None, nsmp=config.NUM_SAMPLE, return_fullfea=True)
        end_points = dict_stop_gradient(end_points)
    if config.SHRINK_BOX:
        end_points['bbox_ins_pred'] = box_shrink(end_points['bbox_ins_pred'], pc)
    group_label_onehot = tf.one_hot(group_label, depth=config.NUM_GROUP, axis=-1) #[B, NUM_POINT, NUM_GROUP]
    seg_label_per_group = tf.multiply(tf.cast(tf.expand_dims(seg_label,-1), tf.float32), group_label_onehot)
    seg_label_per_group = tf.cast(tf.round(tf.divide(tf.reduce_sum(seg_label_per_group, 1),tf.reduce_sum(group_label_onehot, 1)+1e-8)), tf.int32) #[B, NUM_GROUP]

    if 'RPOINTNET' in config.TRAIN_MODULE or mode=='inference':
        SPN_NMS_MAX_SIZE = config.SPN_NMS_MAX_SIZE_TRAINING if mode == "training"\
            else config.SPN_NMS_MAX_SIZE_INFERENCE
        # 3D non maximum suppression - selected_indices: [B, M], spn_rois: [B, M, 6]
        selected_indices = tf.stop_gradient(tf.py_func(nms_3d, [end_points['bbox_ins_pred'], end_points['fb_prob'][:,:,1], config.SPN_PRE_NMS_LIMIT, SPN_NMS_MAX_SIZE, config.SPN_IOU_THRESHOLD, config.SPN_SCORE_THRESHOLD], tf.int32))
        spn_rois = gather_selection(end_points['bbox_ins_pred'], selected_indices, SPN_NMS_MAX_SIZE)

        if mode=='training':
            # Detection target generation - rois: [B, TRAIN_ROIS_PER_IMAGE, 6], target_class_ids: [B, TRAIN_ROIS_PER_IMAGE]
            # target_bbox: [B, TRAIN_ROIS_PER_IMAGE, 6], target_mask_selection_idx: [B, TRAIN_ROIS_PER_IMAGE, NUM_POINT_INS_MASK]
            # target_mask: [B, TRAIN_ROIS_PER_IMAGE, NUM_POINT_INS_MASK], all zero padded
            names = ["rois", "target_class_ids", "target_bbox", "target_mask_selection_idx", "target_mask"]
            outputs = batch_slice(
                [spn_rois, seg_label_per_group, bbox_ins, group_label_onehot, pc],
                lambda v, w, x, y, z: detection_target_gen(v, w, x, y, z, config),
                config.BATCH_SIZE, names=names)
            rois, target_class_ids, target_bbox, target_mask_selection_idx, target_mask = outputs

            # Points cropping - pc_fea_cropped: [B, NUM_ROIS, NUM_POINT_PER_ROI, NFEA]
            # pc_center_cropped: [B, NUM_ROIS, NUM_POINT_PER_ROI, 3]
            # pc_coord_cropped: [B, NUM_ROIS, NUM_POINT_PER_ROI, 3]
            ##### sem fpn fea
            sem_fea_full_l1 = tf_util.conv1d(end_points['sem_fea_full_l1'], 64, 1, padding='VALID', bn=True, is_training=is_training, scope='fpn1', bn_decay=bn_decay)
            sem_fea_full_l2 = tf_util.conv1d(end_points['sem_fea_full_l2'], 64, 1, padding='VALID', bn=True, is_training=is_training, scope='fpn2', bn_decay=bn_decay)
            sem_fea_full_l3 = tf_util.conv1d(end_points['sem_fea_full_l3'], 64, 1, padding='VALID', bn=True, is_training=is_training, scope='fpn3', bn_decay=bn_decay)
            sem_fea_full_l4 = tf_util.conv1d(end_points['sem_fea_full_l4'], 64, 1, padding='VALID', bn=True, is_training=is_training, scope='fpn4', bn_decay=bn_decay)
            pc_fea_cropped, pc_center_cropped, pc_coord_cropped, _ = points_cropping(pc, tf.concat((end_points['entity_fea'], sem_fea_full_l1, sem_fea_full_l2, sem_fea_full_l3, sem_fea_full_l4), -1), 
                end_points['center_pos'], rois, target_mask_selection_idx, config.TRAIN_ROIS_PER_IMAGE, config.NUM_POINT_INS_MASK, config.NORMALIZE_CROP_REGION)

            # Classification and bbox refinement head
            rpointnet_class_logits, rpointnet_class, rpointnet_bbox = classification_head(pc_coord_cropped, 
                tf.concat((pc_fea_cropped, pc_center_cropped), -1), config.NUM_CATEGORY, 
                [128, 256, 512], [256, 256], is_training, bn_decay, 'classification_head')

            # Mask prediction head
            rpointnet_mask = segmentation_head(pc_coord_cropped,
                tf.concat((pc_fea_cropped, pc_center_cropped), -1), config.NUM_CATEGORY,
                [64, 64], [64, 128, 512], [256, 256], is_training, bn_decay, 'segmentation_head')
        elif mode=='inference':
            # rois: [B, NUM_ROIS, 6]
            names = ["rois", "mask_selection_idx"]
            outputs = batch_slice(
                [spn_rois, pc],
                lambda x, y: mask_selection_gen(x, y, SPN_NMS_MAX_SIZE, config, empty_removal=True),
                config.BATCH_SIZE, names=names)
            rois, mask_selection_idx = outputs

            # Points cropping - pc_fea_cropped: [B, NUM_ROIS, NUM_POINT_PER_ROI, NFEA]
            # pc_center_cropped: [B, NUM_ROIS, NUM_POINT_PER_ROI, 3]
            # pc_coord_cropped: [B, NUM_ROIS, NUM_POINT_PER_ROI, 3]
            ##### sem fpn fea
            sem_fea_full_l1 = tf_util.conv1d(end_points['sem_fea_full_l1'], 64, 1, padding='VALID', bn=True, is_training=is_training, scope='fpn1', bn_decay=bn_decay)
            sem_fea_full_l2 = tf_util.conv1d(end_points['sem_fea_full_l2'], 64, 1, padding='VALID', bn=True, is_training=is_training, scope='fpn2', bn_decay=bn_decay)
            sem_fea_full_l3 = tf_util.conv1d(end_points['sem_fea_full_l3'], 64, 1, padding='VALID', bn=True, is_training=is_training, scope='fpn3', bn_decay=bn_decay)
            sem_fea_full_l4 = tf_util.conv1d(end_points['sem_fea_full_l4'], 64, 1, padding='VALID', bn=True, is_training=is_training, scope='fpn4', bn_decay=bn_decay)

            #### generate fb_conf and sem_conf
            # pc: [B, N, 3], fb_prob: [B, nsmp, 2], pc_seed: [B, nsmp, 3] -> fb_prob: [B, N]
            midx = tf.argmin(tf.reduce_sum(tf.square(tf.expand_dims(pc, 2)-tf.expand_dims(end_points['pc_seed'],1)),-1), 2)
            midx_aug = tf.tile(tf.reshape(tf.range(config.BATCH_SIZE, dtype=tf.int64),[-1,1]), [1,config.NUM_POINT])
            midx_aug = tf.stack((tf.reshape(midx_aug, [-1]), tf.reshape(midx, [-1])), 1)
            fb_prob = tf.reshape(tf.gather_nd(end_points['fb_prob'], midx_aug), [config.BATCH_SIZE, config.NUM_POINT, 2])
            fb_prob = fb_prob[:,:,1]
            sem_prob = tf.nn.softmax(end_points['sem_class_logits'], -1) #[B, NUM_POINT, NUM_CATEGORY]

            ##### sem fpn fea
            pc_fea_cropped, pc_center_cropped, pc_coord_cropped, pc_coord_cropped_unnormalized = points_cropping(pc, tf.concat((end_points['entity_fea'], sem_fea_full_l1, sem_fea_full_l2, sem_fea_full_l3, sem_fea_full_l4, tf.expand_dims(fb_prob, -1), sem_prob), -1), 
                end_points['center_pos'], rois, mask_selection_idx, SPN_NMS_MAX_SIZE, config.NUM_POINT_INS_MASK, config.NORMALIZE_CROP_REGION)
            pc_fea_cropped, fb_prob_cropped, sem_prob_cropped = tf.split(pc_fea_cropped, [1024, 1, config.NUM_CATEGORY], -1)
            
            fb_prob_cropped = tf.reduce_mean(tf.squeeze(fb_prob_cropped, -1),-1) # [B, NUM_ROIS]
            end_points['fb_prob_cropped'] = fb_prob_cropped
            sem_prob_cropped = tf.reduce_mean(sem_prob_cropped, 2) # [B, NUM_ROIS, NUM_CATEGORY]

            # Classification and bbox refinement head - rpointnet_class_logits: [B, NUM_ROIS, NUM_CATEGORY]
            # rpointnet_class: [B, NUM_ROIS, NUM_CATEGORY],
            # rpointnet_bbox: [B, NUM_ROIS, NUM_CATEGORY, 6]
            rpointnet_class_logits, rpointnet_class, rpointnet_bbox = classification_head(pc_coord_cropped, 
                tf.concat((pc_fea_cropped, pc_center_cropped), -1), config.NUM_CATEGORY, 
                [128, 256, 512], [256, 256], is_training, bn_decay, 'classification_head')

            midx = tf.argmax(rpointnet_class_logits, -1) # [B, NUM_ROIS]
            midx = tf.stack((tf.range(config.BATCH_SIZE*SPN_NMS_MAX_SIZE, dtype=tf.int64), tf.reshape(midx,[-1])), 1)
            sem_prob_cropped = tf.gather_nd(tf.reshape(sem_prob_cropped, [-1, config.NUM_CATEGORY]), midx)
            sem_prob_cropped = tf.reshape(sem_prob_cropped, [config.BATCH_SIZE, SPN_NMS_MAX_SIZE])
            end_points['sem_prob_cropped'] = sem_prob_cropped

            # Generate detections: [B, DETECTION_MAX_INSTANCES, (center_x, center_y, center_z, l, w, h, class_id, score)]
            detections = batch_slice(
                [rois, rpointnet_class, rpointnet_bbox, pc, fb_prob_cropped, sem_prob_cropped],
                lambda u, v, x, y, w, z: refine_detections(u, v, x, y, w, z, config),
                config.BATCH_SIZE)

            # Re-crop point cloud for mask prediction
            names = ["rois_final", "mask_selection_idx_final"]
            outputs = batch_slice(
                [detections[:,:,:6], pc],
                lambda x, y: mask_selection_gen(x, y, config.DETECTION_MAX_INSTANCES, config, empty_removal=False),
                config.BATCH_SIZE, names=names)
            rois_final, mask_selection_idx_final = outputs
            # Points cropping - pc_fea_cropped_final: [B, DETECTION_MAX_INSTANCES, NUM_POINT_PER_ROI, NFEA]
            # pc_center_cropped_final: [B, DETECTION_MAX_INSTANCES, NUM_POINT_PER_ROI, 3]
            # pc_coord_cropped_final: [B, DETECTION_MAX_INSTANCES, NUM_POINT_PER_ROI, 3]
            ##### sem fpn fea
            pc_fea_cropped_final, pc_center_cropped_final, pc_coord_cropped_final, pc_coord_cropped_final_unnormalized = points_cropping(pc, tf.concat((end_points['entity_fea'], sem_fea_full_l1, sem_fea_full_l2, sem_fea_full_l3, sem_fea_full_l4), -1), 
                end_points['center_pos'], rois_final, mask_selection_idx_final, config.DETECTION_MAX_INSTANCES, config.NUM_POINT_INS_MASK, config.NORMALIZE_CROP_REGION)

            # Mask prediction head
            rpointnet_mask = segmentation_head(pc_coord_cropped_final,
                tf.concat((pc_fea_cropped_final, pc_center_cropped_final), -1), config.NUM_CATEGORY,
                [64, 64], [64, 128, 512], [256, 256], is_training, bn_decay, 'segmentation_head')

            # Unmold segmentation
            rpointnet_mask_selected = select_segmentation(tf.nn.sigmoid(rpointnet_mask), detections[:,:,6])

    # Update end_points
    end_points['group_label'] = group_label
    end_points['seg_label'] = seg_label
    end_points['seg_label_per_group'] = seg_label_per_group #[B, NUM_GROUP]
    end_points['bbox_ins'] = bbox_ins #[B, NUM_GROUP, 6]
    if 'RPOINTNET' in config.TRAIN_MODULE and mode=='training':
        end_points['selected_indices'] = selected_indices #[B, SPN_NMS_MAX_SIZE]
        end_points['spn_rois'] = spn_rois #[B, SPN_NMS_MAX_SIZE, 6]
        end_points['rois'] = rois #[B, NUM_ROIS, 6]
        end_points['target_class_ids'] = target_class_ids #[B, NUM_ROIS]
        end_points['target_bbox'] = target_bbox #[B, NUM_ROIS, 6]
        end_points['target_mask_selection_idx'] = target_mask_selection_idx #[B, NUM_ROIS, NUM_POINT_PER_ROI]
        end_points['target_mask'] = target_mask #[B, NUM_ROIS, NUM_POINT_PER_ROI]
        end_points['rpointnet_class_logits'] = rpointnet_class_logits #[B, NUM_ROIS, NUM_CATEGORY]
        end_points['rpointnet_class'] = rpointnet_class #[B, NUM_ROIS, NUM_CATEGORY]
        end_points['rpointnet_bbox'] = rpointnet_bbox #[B, NUM_ROIS, NUM_CATEGORY, 6]
        end_points['rpointnet_mask'] = rpointnet_mask #[B, NUM_ROIS, NUM_POINT_PER_ROI, NUM_CATEGORY]
    elif mode=='inference':
        end_points['selected_indices'] = selected_indices #[B, SPN_NMS_MAX_SIZE]
        end_points['spn_rois'] = spn_rois #[B, SPN_NMS_MAX_SIZE, 6]
        end_points['rois'] = rois #[B, NUM_ROIS, 6]
        end_points['rpointnet_class_logits'] = rpointnet_class_logits #[B, NUM_ROIS, NUM_CATEGORY]
        end_points['rpointnet_class'] = rpointnet_class #[B, NUM_ROIS, NUM_CATEGORY]
        end_points['rpointnet_bbox'] = rpointnet_bbox #[B, NUM_ROIS, NUM_CATEGORY, 6]
        end_points['detections'] = detections #[B, DETECTION_MAX_INSTANCES, 6+2]
        end_points['rpointnet_mask'] = rpointnet_mask #[B, DETECTION_MAX_INSTANCES, NUM_POINT_PER_ROI, NUM_CATEGORY]
        end_points['rpointnet_mask_selected'] = rpointnet_mask_selected #[B, DETECTION_MAX_INSTANCES, NUM_POINT_PER_ROI]
        end_points['pc_coord_cropped_final_unnormalized'] = pc_coord_cropped_final_unnormalized #[B, DETECTION_MAX_INSTANCES, NUM_POINT_PER_ROI, 3]
    return end_points

def smooth_l1_loss(y_true, y_pred):
    """Implements Smooth-L1 loss.
    y_true and y_pred are typically: [N, 6], but could be any shape.
    """
    diff = tf.abs(y_true - y_pred)
    less_than_one = tf.cast(tf.less(diff, 1.0), tf.float32)
    loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)
    return loss

def get_spn_class_loss(fb_logits, spn_match):
    '''
    Inputs:
       fb_logits: [B, nsmp, 2]
       spn_match: [B, nsmp]
    '''
    # Only postive and negative contribute to loss but not neutral
    fb_logits = tf.reshape(fb_logits, [-1, 2])
    spn_match = tf.reshape(spn_match, [-1])
    valid_mask = tf.not_equal(spn_match, 0)
    fb_logits = tf.boolean_mask(fb_logits, valid_mask)
    spn_match = tf.cast(tf.equal(spn_match, 1), tf.int32)
    spn_match = tf.boolean_mask(spn_match, valid_mask)
    loss = tf.cond(tf.size(spn_match)>0,
        lambda: tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=spn_match, logits=fb_logits)),
        lambda: tf.constant(0.0))

    return loss

def get_rpointnet_class_loss(rpointnet_class_logits, gt_class_ids, roi_valid_mask):
    '''
    Inputs:
       rpointnet_class_logits: [B, NUM_ROIS, NUM_CATEGORY]
       gt_class_ids: [B, NUM_ROIS], zero padded
       roi_valid_mask: [B, NUM_ROIS]
    '''
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gt_class_ids, logits=rpointnet_class_logits)
    loss = tf.multiply(loss, roi_valid_mask)
    loss = tf.divide(tf.reduce_sum(loss), tf.reduce_sum(roi_valid_mask)+1e-8)

    return loss

def get_rpointnet_bbox_loss(gt_bbox, gt_class_ids, pred_bbox, roi_valid_mask, num_category):
    '''
    Inputs:
       gt_bbox: [B, NUM_ROIS, 6]
       gt_class_ids: [B, NUM_ROIS], zero padded
       pred_bbox: [B, NUM_ROIS, NUM_CATEGORY, 6]
       roi_valid_mask: [B, NUM_ROIS]
       num_category: scalar
    '''
    # Only foreground box contribute to the loss
    gt_bbox = tf.reshape(gt_bbox, [-1, 6])
    gt_class_ids = tf.reshape(gt_class_ids, [-1])
    pred_bbox = tf.reshape(pred_bbox, [-1, num_category, 6])
    roi_valid_mask = tf.reshape(roi_valid_mask, [-1])

    gt_selected_indices = tf.where(tf.logical_and(tf.greater(roi_valid_mask, 0), tf.greater(gt_class_ids, 0)))[:,0]
    gt_selected_indices = tf.cast(gt_selected_indices, tf.int32)
    pred_selected_indices = tf.concat((tf.reshape(gt_selected_indices, [-1,1]),
        tf.reshape(tf.gather(gt_class_ids, gt_selected_indices), [-1,1])), axis=1)

    gt_bbox = tf.gather(gt_bbox, gt_selected_indices, axis=0)
    pred_bbox = tf.gather_nd(pred_bbox, pred_selected_indices)

    loss = tf.cond(tf.size(gt_bbox)>0,
        lambda: tf.reduce_mean(tf.reduce_sum(smooth_l1_loss(y_true=gt_bbox, y_pred=pred_bbox),1),0),
        lambda: tf.constant(0.0))

    return loss

def get_rpointnet_mask_loss(gt_masks, gt_class_ids, pred_masks, roi_valid_mask, num_category, num_point_per_roi):
    '''
    Inputs:
       gt_masks: [B, NUM_ROIS, NUM_POINT_PER_ROI]
       gt_class_ids: [B, NUM_ROIS], zero padded
       pred_masks: [B, NUM_ROIS, NUM_POINT_PER_ROI, NUM_CATEGORY]
       roi_valid_mask: [B, NUM_ROIS]
       num_category: scalar
       num_point_per_roi: scalar
    '''
    # Only foreground box contribute to the loss
    gt_masks = tf.reshape(gt_masks, [-1, num_point_per_roi])
    gt_class_ids = tf.reshape(gt_class_ids, [-1])
    pred_masks = tf.reshape(pred_masks, [-1, num_point_per_roi, num_category])
    pred_masks = tf.transpose(pred_masks, perm=[0,2,1])
    roi_valid_mask = tf.reshape(roi_valid_mask, [-1])

    gt_selected_indices = tf.where(tf.logical_and(tf.greater(roi_valid_mask, 0), tf.greater(gt_class_ids, 0)))[:,0]
    gt_selected_indices = tf.cast(gt_selected_indices, tf.int32)
    pred_selected_indices = tf.concat((tf.reshape(gt_selected_indices, [-1,1]),
        tf.reshape(tf.gather(gt_class_ids, gt_selected_indices), [-1,1])), axis=1)

    gt_masks = tf.gather(gt_masks, gt_selected_indices, axis=0) # [N, NUM_POINT_PER_ROI]
    gt_masks = tf.cast(gt_masks, tf.float32)
    pred_masks = tf.gather_nd(pred_masks, pred_selected_indices) # [N, NUM_POINT_PER_ROI]

    loss = tf.cond(tf.size(gt_masks)>0,
        lambda: tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=gt_masks, logits=pred_masks)),
        lambda: tf.constant(0.0))

    return loss

def get_loss(end_points, config, alpha, smpw, mode='training'):
    batch_size = config.BATCH_SIZE
    nsmp_ins = config.NUM_POINT_INS
    bbox_size = tf.reduce_max(end_points['pc_ins_centered_seed'], axis=2, keep_dims=True)-tf.reduce_min(end_points['pc_ins_centered_seed'], axis=2, keep_dims=True)
    radius = 1e-8 + tf.sqrt(tf.reduce_sum(tf.square(bbox_size/2), axis=-1, keep_dims=True)) # [B, nsmp, 1, 1]
    shift_gt_seed = end_points['pc_ins_center_seed']-tf.expand_dims(end_points['pc_seed'],2) # [B, nsmp, 1, 3]
    shift_dist = tf.sqrt(tf.reduce_sum(tf.square(shift_gt_seed), 3, keep_dims=True)+1e-8)
    shift_gt_seed_normalized_4d = tf.concat((tf.divide(shift_gt_seed, shift_dist), tf.divide(shift_dist, radius)), -1)
    shift_pred_seed_normalized_4d = tf.concat((tf.expand_dims(end_points['shift_pred_seed_4d'],2)[:,:,:,:3], tf.divide(tf.expand_dims(end_points['shift_pred_seed_4d'],2)[:,:,:,3:], radius)), -1)

    # Fg/Bg loss, spn_match: [B, nsmp]
    fb_score_gt = tf.squeeze(gather_selection(tf.expand_dims(end_points['seg_label'],-1), end_points['ind_seed'], end_points['ind_seed'].get_shape()[1].value),-1)
    fb_score_gt = tf.cast(tf.greater(fb_score_gt,0), tf.float32) # [B, nsmp]
    spn_match = batch_slice(
        [end_points['bbox_ins_pred'], fb_score_gt, tf.cast(tf.greater(end_points['seg_label_per_group'],0),tf.float32), end_points['bbox_ins']],
        lambda w, x, y, z: spn_target_gen(w, x, y, z),
        batch_size, names=["spn_match"])
    spn_match = tf.stop_gradient(tf.cast(spn_match, tf.int32))
    end_points['spn_match'] = spn_match
    spn_class_loss = get_spn_class_loss(end_points['fb_logits'], spn_match)

    # # Reconstruction loss
    pc_ins_pred = end_points['pc_ins_pred']
    pc_ins_pred_normalized = tf.reshape(tf.div(pc_ins_pred, radius), [-1, nsmp_ins, 3]) # [B*nsmp, nsmp_ins, 3]
    pc_ins_gt_normalized = tf.reshape(tf.div(end_points['pc_ins_centered_seed']+shift_gt_seed, radius), [-1, nsmp_ins, 3]) # [B*nsmp, nsmp_ins, 3]
    recon_valid_mask = tf.reshape(fb_score_gt, [-1])
    recon_valid_mask = tf.stop_gradient(recon_valid_mask)
    dists_forward,_,dists_backward,_ = tf_nndistance.nn_distance(pc_ins_pred_normalized, pc_ins_gt_normalized)
    recons_loss = tf.reduce_mean(dists_forward+dists_backward, axis=-1) # B*nsmp
    recons_loss = tf.divide(tf.reduce_sum(tf.multiply(recons_loss, recon_valid_mask)),
        tf.reduce_sum(recon_valid_mask)+1e-8)

    # Shift loss
    shift_loss = tf.reduce_sum(smooth_l1_loss(shift_gt_seed_normalized_4d, shift_pred_seed_normalized_4d), axis=-1)
    shift_loss = tf.reduce_sum(tf.multiply(tf.reshape(shift_loss, [-1]), recon_valid_mask))
    shift_loss = tf.divide(shift_loss, tf.reduce_sum(recon_valid_mask)+1e-8)

    # Sem loss
    ind_sem = end_points['ind_sem']
    nsmp_sem = ind_sem.get_shape()[1].value
    sem_labels = end_points['seg_label']
    ind_sem_aug = tf.tile(tf.reshape(tf.range(batch_size),[-1, 1]), [1, nsmp_sem])
    ind_sem_aug = tf.concat( (tf.reshape(ind_sem_aug, [-1, 1]), tf.reshape(ind_sem, [-1, 1])), 1 )
    sem_labels = tf.reshape(tf.gather_nd(sem_labels, ind_sem_aug), [batch_size, nsmp_sem])
    smpw = tf.reshape(tf.gather_nd(smpw, ind_sem_aug), [batch_size, nsmp_sem])
    sem_labels = tf.cast(sem_labels, tf.int32)
    sem_loss = tf.losses.sparse_softmax_cross_entropy(labels=sem_labels, logits=end_points['sem_class_logits'], weights=smpw)
    end_points['sem_labels'] = sem_labels

    # KL loss
    mean = end_points['mean'] # [B, nsmp, 256]
    log_var = end_points['log_var']
    cmean = end_points['cmean']
    clog_var = end_points['clog_var']
    kl_loss = 0.5 * tf.reduce_mean( log_var - clog_var + (tf.exp(clog_var) + (mean-cmean)**2)/tf.exp(log_var) - 1.0, 2) 
    kl_loss = tf.divide(tf.reduce_sum(tf.multiply(tf.reshape(kl_loss, [-1]), recon_valid_mask)),
        tf.reduce_sum(recon_valid_mask)+1e-8)

    if 'RPOINTNET' in config.TRAIN_MODULE and mode=='training':
        # rpointnet classification loss
        roi_valid_mask = tf.cast(tf.not_equal(tf.reduce_sum(tf.abs(end_points['rois']), axis=-1),0), tf.float32)
        rpointnet_class_loss = get_rpointnet_class_loss(end_points['rpointnet_class_logits'], end_points['target_class_ids'], roi_valid_mask)

        # rpointnet bbox loss
        rpointnet_bbox_loss = get_rpointnet_bbox_loss(end_points['target_bbox'], end_points['target_class_ids'], end_points['rpointnet_bbox'], roi_valid_mask, config.NUM_CATEGORY)

        # rpointnet mask loss
        rpointnet_mask_loss = get_rpointnet_mask_loss(end_points['target_mask'], end_points['target_class_ids'], end_points['rpointnet_mask'], roi_valid_mask, config.NUM_CATEGORY, config.NUM_POINT_INS_MASK)
    
    if 'SPN' in config.TRAIN_MODULE:
        loss = kl_loss * alpha + recons_loss + shift_loss + spn_class_loss + sem_loss
        if 'RPOINTNET' in config.TRAIN_MODULE and mode=='training':
            loss += rpointnet_class_loss + rpointnet_bbox_loss + rpointnet_mask_loss
    elif 'RPOINTNET' in config.TRAIN_MODULE and mode=='training':
        loss = rpointnet_class_loss + rpointnet_bbox_loss + rpointnet_mask_loss
    else:
        loss = tf.constant(0.0)


    # Store end_points
    end_points['spn_class_loss'] = spn_class_loss
    end_points['recons_loss'] = recons_loss
    end_points['shift_loss'] = shift_loss
    end_points['sem_loss'] = sem_loss
    end_points['kl_loss'] = kl_loss
    if 'RPOINTNET' in config.TRAIN_MODULE and mode=='training':
        end_points['rpointnet_class_loss'] = rpointnet_class_loss
        end_points['rpointnet_bbox_loss'] = rpointnet_bbox_loss
        end_points['rpointnet_mask_loss'] = rpointnet_mask_loss
    end_points['loss'] = loss
    
    return loss, end_points


if __name__=='__main__':
    myconfig = config.Config()
    with tf.Graph().as_default():
        pc_pl, pc_ins_pl, group_label_pl, group_indicator_pl, seg_label_pl, bbox_ins_pl = placeholder_inputs(myconfig)
        end_points = rpointnet(pc_pl, pc_ins_pl, group_label_pl, group_indicator_pl, seg_label_pl, bbox_ins_pl, myconfig, tf.constant(True), bn_decay=None)
        loss, end_points = get_loss(end_points, myconfig, 1.0)
        print(end_points)
