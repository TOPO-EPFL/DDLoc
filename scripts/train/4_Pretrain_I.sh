#### Step4: Train inpaint module

# pretrain with naive mixed data
python train_I.py --exp_dir 'checkpoints/urban' --path_to_real 'your absolute path to real urban dataset' \
  --path_to_syn 'your absolute path to synthetic urban dataset' \
  --batch_size=2 --total_epoch_num=150 --isTrain --save_steps=10 --start_epoch=0 --img_normalize='pure'

# finetune with 1 to 1 matched data
python train_I.py --exp_dir 'checkpoints/urban' --path_to_real 'your absolute path to real urban dataset' \
  --path_to_syn 'your absolute path to paired synthetic urban dataset' \
  --batch_size=2 --total_epoch_num=200 --isTrain --save_steps=10 --start_epoch=150 --img_normalize='pure' --data_ispaired
