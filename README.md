# GSPN: *Generative Shape Proposal Network for 3D Instance Segmentation in Point Cloud*
Created by <a href="https://cs.stanford.edu/~ericyi/" target="_blank">Li (Eric) Yi</a>, Wang Zhao, <a href="http://ai.stanford.edu/~hewang/" target="_blank">He Wang</a>, <a href="https://mhsung.github.io/" target="_blank">Minhyuk Sung</a>, <a href="http://geometry.stanford.edu/member/guibas/" target="_blank">Leonidas J. Guibas</a>.

### Citation
If you find our work useful in your research, please consider citing:

        @article{yi2018gspn,
          title={Gspn: Generative shape proposal network for 3d instance segmentation in point cloud},
          author={Yi, Li and Zhao, Wang and Wang, He and Sung, Minhyuk and Guibas, Leonidas},
          journal={arXiv preprint arXiv:1812.03320},
          year={2018}
        }
        
### Introduction
This work is based on our CVPR'19 paper. You can find arXiv version of the paper <a href="https://arxiv.org/abs/1812.03320">here</a>. We introduce a 3D object proposal approach named Generative Shape Proposal Network (GSPN) for instance segmentation in point cloud data. We incorporate GSPN into a novel 3D instance segmentation framework named Region-based PointNet (R-PointNet) which allows flexible proposal refinement and instance segmentation generation.

In this repository we release code and pre-trained model for both GSPN and R-PointNet.

### Usage
We provide a step-by-step usage instruction from data processing to network evaluation on the <a href="http://www.scan-net.org/">ScanNet</a> dataset.

1. Compiling the TF operators

        cd tf_ops
        . ./tf_all_compile.sh
        
2. Data pre-processing: convert ScanNet into downsampled point cloud for fast training and evaluation

        python data_prep.py
        python dataset.py
        
3. Two-step training: we train GSPN and R-PointNet separately in two different stages

        python train.py --train_module SPN --log_dir log_spn
        python train.py --train_module RPOINTNET --log_dir log_rpointnet --restore_model_path log_spn/model.ckpt --restore_scope shape_proposal_net
        
4. Evaluation on the validation set: we evaluation on downsampled point cloud from ScanNet validation scenes, where we first generate predictions and then evaluate based upon the official code provided by <a href="https://github.com/ScanNet/ScanNet">ScanNet Benchmark</a>

        python test.py --model_path log_rpointnet/model.ckpt
        cd ./external/ScanNet/BenchmarkScripts/3d_evaluation
        python evaluate_semantic_instance.py --pred_path root_dir/eva/pred --gt_path root_dir/eva/gt --output_file root_dir/eva/pred/semantic_instance_evaluation.txt
        
### Pre-trained model
The pretrained R-PointNet could be downloaded <a href="https://shapenet.cs.stanford.edu/ericyi/rpointnet_pretrained.zip">here</a> (57Mb).

### License
Our code is released under MIT License (see LICENSE file for details).
