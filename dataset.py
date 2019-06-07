import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'tf_ops/sampling'))
import copy
import io_util
import tensorflow as tf
import tf_sampling
import numpy as np
import scipy.io as sio

class ScanNetDataset():
    def __init__(self, src_data_path, list_path, cache_path, npoint=18000, npoint_ins=512, is_augment=False, permute_points=True):
        '''
            src_data_path: path to ScanNet
            list_path: a plain txt file containing all the file names
            cache_path: path to the cached files
            npoint: number of sampled points per scene
            npoint_ins: number of sampled points per instance
        '''
        self.npoint = npoint
        self.npoint_ins = npoint_ins
        self.ngroup = 0
        self.is_augment = is_augment
        self.permute_points = permute_points
        self.file_list = io_util.read_txt(list_path)
        self.data_list = {}
        if os.path.exists(cache_path):
            #### collect the processed scene data
            self.data_list = np.load(cache_path)['data_list'].item()
            #### collect the number of instances per scene
            self.ngroup = np.load(cache_path)['ngroup'].item()
        else:
            self.cache_file(src_data_path, cache_path)

    def cache_file(self, src_data_path, cache_path):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.device('/gpu:0'):
            pc_tf = tf.placeholder(tf.float32)
            ind_tf = tf_sampling.farthest_point_sample(self.npoint, pc_tf)
            ind_ins_tf = tf_sampling.farthest_point_sample(self.npoint_ins, pc_tf)
        sess = tf.Session(config=config)
        nfile = len(self.file_list)
        #### valid class ids defined in ScanNet
        VALID_CLASS_IDS = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
        target_sem_idx = np.arange(40)
        count = 0
        for i in range(40):
            if i in VALID_CLASS_IDS:
                count += 1
                target_sem_idx[i] = count
            else:
                target_sem_idx[i] = 0
        for index in range(nfile):
            print(np.float32(index)/nfile)
            curpc_color = io_util.read_color_ply(os.path.join(src_data_path, 'mesh2', self.file_list[index]+'.ply'))
            curpc = curpc_color[:,:3].astype(np.float32)
            curcolor = curpc_color[:,3:].astype(np.float32)/255.0
            curgroup = io_util.read_label_txt(os.path.join(src_data_path, 'label2', 'group_'+self.file_list[index]+'.txt')).astype(np.int32)
            curseg = io_util.read_label_txt(os.path.join(src_data_path, 'label2', 'sem_'+self.file_list[index]+'.txt')).astype(np.int32)
            curseg[curseg>=40] = 0
            curseg[curseg<0] = 0
            curseg = self.changem(curseg, np.arange(40), target_sem_idx)
            curngroup = np.max(curgroup)+1

            #### collect instance information
            valid_group_indicator = np.zeros(curngroup)
            target_idx = np.zeros(1+curngroup)
            count = 0
            for i in range(curngroup):
                if np.sum(curgroup==i)==0:
                    target_idx[i+1] = 0
                elif np.round(np.mean(curseg[curgroup==i])).astype('int32')!=0:
                    valid_group_indicator[i] = 1
                    count += 1
                    target_idx[i+1] = count
                else:
                    target_idx[i+1] = 0
            curgroup = self.changem(curgroup, np.arange(-1, curngroup), target_idx).astype('int32')
            curgroup[curgroup<0] = 0
            curngroup = 1+np.sum(valid_group_indicator).astype('int32') # group zero is background

            #### resample each scene to a fix number of points
            if curngroup>self.ngroup:
                self.ngroup = curngroup
            if self.npoint<curpc.shape[0]:
                choice = sess.run(ind_tf, feed_dict={pc_tf: np.expand_dims(curpc,0)})[0]
                pc = curpc[choice,:]
                color = curcolor[choice,:]
                group_label = curgroup[choice]
                seg_label = curseg[choice]
            elif self.npoint==curpc.shape[1]:
                pc = copy.deepcopy(curpc)
                color = copy.deepcopy(curcolor)
                group_label = copy.deepcopy(curgroup)
                seg_label = copy.deepcopy(curseg)
            else:
                choice = np.random.choice(curpc.shape[0], self.npoint - curpc.shape[0])
                pc = np.concatenate((curpc,curpc[choice,:]), 0)
                color = np.concatenate((curcolor,curcolor[choice,:]), 0)
                group_label = np.concatenate((curgroup, curgroup[choice]), 0)
                seg_label = np.concatenate((curseg, curseg[choice]), 0)

            #### resample each instance to a fix number of points
            pc_ins = np.zeros((curngroup, self.npoint_ins, 3), dtype=np.float32)
            for j in range(1,curngroup):
                curins = curpc[curgroup==j,:]
                if self.npoint_ins<curins.shape[0]:
                    choice = sess.run(ind_ins_tf, feed_dict={pc_tf: np.expand_dims(curins,0)})[0]
                    pc_ins[j,:,:] = curins[choice,:]
                elif self.npoint_ins==curins.shape[0]:
                    pc_ins[j,:,:] = copy.deepcopy(curins)
                else:
                    choice = np.random.choice(curins.shape[0], self.npoint_ins - curins.shape[0])
                    pc_ins[j,:,:] = np.concatenate((curins,curins[choice,:]), 0)

            #### data tuple for each scene
            #### group_label indicates instance label
            #### seg_label indicates semantic label
            self.data_list[index] = (pc, color, group_label, seg_label, pc_ins, curngroup)

        np.savez_compressed(cache_path, data_list=self.data_list, ngroup=self.ngroup)

    def gen_rotation_matrix(self):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, sinval, 0],
                                    [-sinval, cosval, 0],
                                    [0, 0, 1]])
        return rotation_matrix

    def changem(self, input_array, source_idx, target_idx):
        mapping = {}
        for i, sidx in enumerate(source_idx):
            mapping[sidx] = target_idx[i]
        input_array = np.array([mapping[i] for i in input_array])
        return input_array

    def __getitem__(self, index):
        '''
        Return:
            pc: [npoint, 3], point cloud of the whole scene, in world coord system
            color: [npoint, 3], rgb for each point
            pc_ins_full: [ngroup, npoint_ins, 3], point cloud of all instances, in world coord system, all zero for background instance
            group_label: [npoint], instance label, 0 means background
            group_indicator: [ngroup], indicates which groups are valid, usually the first few
            seg_label: [npoint], semantic label, 0 means background class
        '''
        pc, color, group_label, seg_label, pc_ins, curngroup = self.data_list[index]

        #### randomly permute the point order
        if self.permute_points:
            ridx = np.random.permutation(pc.shape[0])
            pc = copy.deepcopy(pc[ridx,:])
            color = copy.deepcopy(color[ridx,:])
            seg_label = copy.deepcopy(seg_label[ridx])
            group_label = copy.deepcopy(group_label[ridx])
        else:
            pc = copy.deepcopy(pc)
            color = copy.deepcopy(color)
            seg_label = copy.deepcopy(seg_label)
            group_label = copy.deepcopy(group_label)

        #### pad instance to a maximum instance number
        group_indicator = np.zeros((self.ngroup), dtype=np.int32)
        group_indicator[:curngroup] = 1
        pc_ins_full = np.zeros((self.ngroup, self.npoint_ins, 3), dtype=np.float32)
        pc_ins_full[:curngroup,:,:] = pc_ins

        #### augmenting the input scene with random rotation around z-axis and random translation
        if self.is_augment:
            R = self.gen_rotation_matrix()
            pc = np.matmul(pc, R)
            pc_ins_full = np.reshape(np.matmul(np.reshape(pc_ins_full, [-1, 3]), R), pc_ins_full.shape)

            # aug translation
            t = np.random.normal(0,1,[1,3])
            pc += t
            pc_ins_full += np.reshape(t, [1,1,3])

        #### compute axis aligned bounding box for all instances
        bbox_ins_full = np.zeros((self.ngroup, 6), dtype=np.float32)
        bbox_ins_full[:, :3] = (np.max(pc_ins_full,1)+np.min(pc_ins_full,1))/2
        bbox_ins_full[:, 3:] = np.max(pc_ins_full,1)-np.min(pc_ins_full,1)

        return pc, color, pc_ins_full, group_label, group_indicator, seg_label, bbox_ins_full


    def __len__(self):
        return len(self.file_list)

if __name__ == '__main__':
    npoint = 18000 # number of sampled points per scene
    npoint_ins = 512 # number of sampled points per instance
    if not os.path.exists(os.path.join(BASE_DIR, 'data/cache')):
        os.makedirs(os.path.join(BASE_DIR, 'data/cache'))
    src_data_path = os.path.join(BASE_DIR, 'data/scannet_preprocessed')
    train_list = os.path.join(BASE_DIR, 'data/scannet/scannet_train.txt')
    val_list = os.path.join(BASE_DIR, 'data/scannet/scannet_val.txt')
    train_cache = os.path.join(BASE_DIR, 'data/cache/train_%d_%d.npz'%(npoint, npoint_ins))
    val_cache = os.path.join(BASE_DIR, 'data/cache/val_%d_%d.npz'%(npoint, npoint_ins))
    trainDataset = ScanNetDataset(src_data_path, train_list, train_cache, npoint=npoint, npoint_ins=npoint_ins, is_augment=True)
    print(len(trainDataset))
    pc, color, pc_ins_full, group_label, group_indicator, seg_label, bbox_ins_full = trainDataset[0]
    valDataset = ScanNetDataset(src_data_path, val_list, val_cache, npoint=npoint, npoint_ins=npoint_ins, is_augment=False)
    print(len(valDataset))
    pc_val, color_val, pc_ins_full_val, group_label_val, group_indicator_val, seg_label_val, bbox_ins_full_val = valDataset[0]
