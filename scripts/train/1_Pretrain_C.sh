#### Step1: Pretrain coordinate regressor

python train_C.py --exp_dir 'checkpoints/urban' --path_to_real 'your absolute path to real urban dataset' \
  --path_to_syn 'your absolute path to synthetic urban dataset' \
  --batch_size=60 --total_epoch_num=1500 --isTrain --start_epoch=0 --isdownsample --img_normalize='urban'
