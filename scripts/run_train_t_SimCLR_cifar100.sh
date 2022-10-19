# parameters
model=resnet50
epochs=1000
bsz=1024
lr=0.4
temp=8
method=t-SimCLR
decay=0.0001
dataset=cifar100
timestamp=`date '+%s'`

# preparation
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd ${SHELL_FOLDER}/../
if [ ! -d "./results" ]; then
    mkdir ./results
fi

# train
python train.py --dataset=${dataset} --epochs=${epochs} --model=${model} --batch_size=${bsz} --learning_rate=${lr} --temp=${temp} --weight_decay=${decay} --cosine --method=${method} | tee ./results/"${timestamp}_${dataset}_${method}_${model}_epochs_${epochs}_bsz_${bsz}_lr_${lr}_temp_${temp}_decay_${decay}".txt