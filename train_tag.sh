DEVICE_ID=3
CONFIG_FILE=./logs/baseline_tag.yaml
python train.py --config_file=${CONFIG_FILE} --device_id=${DEVICE_ID} --k=0
#python train.py --config_file=${CONFIG_FILE} --device_id=${DEVICE_ID} --k=1
#python train.py --config_file=${CONFIG_FILE} --device_id=${DEVICE_ID} --k=2
#python train.py --config_file=${CONFIG_FILE} --device_id=${DEVICE_ID} --k=3
#python train.py --config_file=${CONFIG_FILE} --device_id=${DEVICE_ID} --k=4