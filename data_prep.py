import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'tf_ops/sampling'))
import json
import numpy as np
import tensorflow as tf
import tf_sampling
import io_util


def collect_label(labelPath, scan):
    aggregation = os.path.join(labelPath, scan, scan+'.aggregation.json')
    segs = os.path.join(labelPath, scan, scan+'_vh_clean_2.0.010000.segs.json')
    sem = os.path.join(labelPath, scan, scan+'_vh_clean_2.labels.ply')

    # Load all labels
    fid = open(aggregation,'r')
    aggreData = json.load(fid)
    fid = open(segs,'r')
    segsData = json.load(fid)
    _, semLabel = io_util.read_label_ply(sem)

    # Convert segments to normal labels
    segGroups = aggreData['segGroups']
    segIndices = np.array(segsData['segIndices'])

    # outGroups is the output instance labels
    outGroups = np.zeros(np.shape(segIndices)) - 1

    for j in range(np.shape(segGroups)[0]):
        segGroup = segGroups[j]['segments']
        objectId = segGroups[j]['objectId']
        for k in range(np.shape(segGroup)[0]):
            segment = segGroup[k]
            ind = np.where(segIndices==segment)
            if all(outGroups[ind] == -1) != True:
                print('Error!')
            outGroups[ind] = objectId

    semLabel = np.array(map(int, semLabel))
    outGroups = np.array(map(int, outGroups))
    
    return semLabel, outGroups


if __name__ == '__main__':
    #### datasetPath: ./data/scannet
    #### outPath: ./data/scannet_preprocessed
    assert len(sys.argv)==3, 'Incorrect Number of Arguments'
    datasetPath = sys.argv[1]
    outPath = sys.argv[2]
    trainMeshPath = os.path.join(datasetPath, 'mesh/scans')
    trainLabelPath = os.path.join(datasetPath, 'label/scans')
    trainList = os.listdir(trainMeshPath)

    # Create output path
    if not os.path.exists(os.path.join(outPath, 'mesh/scans')):
        os.makedirs(os.path.join(outPath, 'mesh/scans'))
    if not os.path.exists(os.path.join(outPath, 'label/scans')):
        os.makedirs(os.path.join(outPath, 'label/scans'))

    # Downsample each scene and save the data to outPath
    nSample = 30000
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.device('/gpu:0'):
        points = tf.placeholder(tf.float32)
        ind = tf_sampling.farthest_point_sample(nSample, points)
    sess = tf.Session(config=config)

    for i, scan in enumerate(trainList):
        scene = np.array(io_util.read_color_ply(os.path.join(trainMeshPath, scan, scan+'_vh_clean_2.ply')))
        sem_label, ins_label = collect_label(trainLabelPath, scan)
        if np.shape(scene)[0] <= nSample:
            output = scene
            out_sem = sem_label
            out_ins = ins_label
        else:
            xyz = scene[:,:3]
            pc = np.expand_dims(xyz, axis=0)
            idx = sess.run(ind, feed_dict={points: pc})
            idx = idx[0,:]
            output = scene[idx]
            out_sem = sem_label[idx]
            out_ins = ins_label[idx]
 
        io_util.write_color_ply(output, os.path.join(outPath, 'mesh/scans', scan+'.ply'))
        io_util.write_label_txt(out_sem, os.path.join(outPath, 'label/scans', 'sem_'+scan+'.txt'))
        io_util.write_label_txt(out_ins, os.path.join(outPath, 'label/scans', 'group_'+scan+'.txt'))
        print('Scene ' + str(i) + ' done!')

