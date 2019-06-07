import tensorflow as tf
import numpy as np
from tf_grouping import query_ball_point, group_point, group_maxpool

class GroupPointTest(tf.test.TestCase):
  def test(self):
    pass

  def test_grad(self):
    with tf.device('/gpu:1'):
      points = tf.constant(np.random.random((4,256,8)).astype('float32'))
      print (points)
      xyz1 = tf.constant(np.random.random((4,256,3)).astype('float32'))
      xyz2 = tf.constant(np.random.random((4,256,3)).astype('float32'))
      radius = 0.3 
      nsample = 256
      idx, pts_cnt = query_ball_point(radius, nsample, xyz1, xyz2)
      grouped_points = group_point(points, idx)
      #grouped_points, max_idx = group_maxpool(points, idx)
      print (grouped_points)

    with self.test_session():
      print ("---- Going to compute gradient error")
      err = tf.test.compute_gradient_error(points, (4,256,8), grouped_points, (4,256,256,8))
      #err = tf.test.compute_gradient_error(points, (1,8,4), grouped_points, (1,2,4))
      print (err)
      self.assertLess(err, 1e-4) 

if __name__=='__main__':
  tf.test.main() 
