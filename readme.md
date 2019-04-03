code accompanying the paper `RegularFace: Deep Face Recognition via Exclusive Regularization`.

### Training
1. Download the **CASIA-webface** dataset (you may request for access);
2. Preprocess the dataset with code provided in <https://github.com/wy1iu/sphereface/tree/master/preprocess/code>;
3. Train baseline methods (softmax, NormFace) with `python -m torch.distributed.launch --master_port 8889 --nproc_per_node=$GPUs pytorch/train_baseline.py`
4. (optionally) you can train baseline methods (softmax, NormFace) with `python -m torch.distributed.launch --master_port 8889 --nproc_per_node=$GPUs pytorch/train_sample_exclusive.py`.

