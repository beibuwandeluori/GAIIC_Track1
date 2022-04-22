DEVICE_ID=4
CONFIG_FILE=./logs/baseline_text.yaml
python train.py --config_file=${CONFIG_FILE} --device_id=${DEVICE_ID} --k=0
#python train.py --config_file=${CONFIG_FILE} --device_id=${DEVICE_ID} --k=1
#python train.py --config_file=${CONFIG_FILE} --device_id=${DEVICE_ID} --k=2
#python train.py --config_file=${CONFIG_FILE} --device_id=${DEVICE_ID} --k=3
#python train.py --config_file=${CONFIG_FILE} --device_id=${DEVICE_ID} --k=4