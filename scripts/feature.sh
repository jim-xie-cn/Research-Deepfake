export CUDA_VISIBLE_DEVICES=1
nohup python -u feature.py --action=common> ./logs/common.log 2>&1 &
nohup python -u feature.py --action=mfs> ./logs/mfs.log 2>&1 &
nohup python -u feature.py --action=lac> ./logs/lac.log 2>&1 &
