#### Generate translated image for step 6 of finetune the coordinate regressor

# remember to copy other data ('init','poses','calibration') to this dir if you want to generate train data for step 6
python generate_translated.py --exp_dir 'checkpoints/urban' --path_to_real 'your absolute path to real urban dataset' \
  --path_to_translate 'your absolute path where you want to store the translated real urban dataset generated by ARC' 