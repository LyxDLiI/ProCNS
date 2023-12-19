
# FAZ
python -u test_WSL.py --img_class faz --num_classes 2 --in_chns 1 --root_path ../data --img_class faz --exp faz/ --sup_type keypoint --model unet

# ODOC
python -u test_WSL.py --img_class odoc --num_classes 3 --in_chns 3 --root_path ../data --img_class odoc --exp odoc/ --sup_type scribble --model unet

# Polyp
python -u test_WSL.py --img_class polyp --num_classes 2 --in_chns 3 --root_path ../data --img_class polyp --exp polyp/ --sup_type scribble --model unet

