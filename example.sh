HOME_DIR=$PWD
# Pose initialization
cd $HOME_DIR/ObjTracker
python run_corres.py --config_path ./configs/custom_shoes.yaml
# Coarse Shape reconstruction
cd $HOME_DIR/instant-nsr-pl
python launch.py --config ./configs/neuswotex-hoi-custom_shoes-barf-normal-corres.yaml --gpu 0 --train
# Refine poses
# TODO: update the refine poses part with outlier voting
cd $HOME_DIR/ObjTracker
python refine_poses.py --config_path ./configs/custom_shoes.yaml
# Final reconstruction
cd $HOME_DIR/instant-nsr-pl
python launch.py --config ./configs/colorneus-hoi-custom_shoes-barf-normal-corres-refine.yaml --gpu 0 --train