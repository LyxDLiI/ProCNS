# Ours
# #FAZ
python Unet_ProCNS_Ours.py --root_path ../data --exp faz/Unet_ProCNS --sup_type keypoint --model unet --max_iterations 30000 --batch_size 12 --in_chns 1 --eval_interval 1 --num_classes 2 --base_lr 0.01 --gpus 0 --img_class faz --alpha 0.2 --thr_conf 0.5 --img_size 256 --r_threshold 0.95 --epoch_update_interval 1 --thr_epoch 100 --thr_conf_correction 0.5 --warm_up_epoch 1

# #ODOC
python Unet_ProCNS_Ours.py --root_path ../data --exp odoc/Unet_ProCNS --sup_type scribble --model unet --max_iterations 30000 --batch_size 12 --in_chns 3 --eval_interval 1 --num_classes 3 --base_lr 0.01 --gpus 2 --img_class odoc --alpha 0.2 --thr_conf 0.3 --img_size 384 --r_threshold 0.95 --epoch_update_interval 1 --thr_epoch 400 --thr_conf_correction 0.3 --warm_up_epoch 1

# Polyp
python Unet_ProCNS_Ours.py --root_path ../data --exp polyp/Unet_ProCNS --sup_type block --model unet --max_iterations 60000 --batch_size 12 --in_chns 3 --eval_interval 1 --num_classes 2 --base_lr 0.01 --gpus 2 --img_class polyp --alpha 0.2 --thr_conf 0.5 --img_size 384 --r_threshold 0.95 --epoch_update_interval 1 --thr_epoch 100 --thr_conf_correction 0.5 --warm_up_epoch 1
