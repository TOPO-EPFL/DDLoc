#### Step5: Jointly train attention module and coordinate regressor with paired data

python train_joint_A_C.py --exp_dir 'checkpoints/urban' --path_to_real 'your absolute path to real urban dataset' \
  --path_to_syn 'your absolute path to paired synthetic urban dataset' \
  --batch_size=4 --total_epoch_num=100 --isTrain --start_epoch=0 --isdownsample --img_normalize='urban' --data_ispaired --rho=0.9 
