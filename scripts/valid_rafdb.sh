IDX=0

IFS=',' read -r -a array <<< "$IDX"
len_node=${#array[@]}

export PYTHONPATH=$PYTHONPATH:./

output_dir=./checkpoints/ckpts/RAF-DB
output_eval_dir=${output_dir}/eval

if [ ! -f "${output_dir}/config.json" ]; then
    echo "skip eval: ${output_dir}/config.json not found (run scripts/train_rafdb.sh first)"
    exit 1
fi

if [ -d ${output_eval_dir} ];then
    echo "dir already exists"
else
    mkdir ${output_eval_dir}
fi

CUDA_VISIBLE_DEVICES=$IDX OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=$len_node --master_port=25003 \
    utils/validfaceEMO.py \
    --model-name ${output_dir} \
    --question-file data_list/test/raf-db.txt \
    --output-dir ${output_eval_dir} \
    --conv-format facetask  2>&1 | tee ${output_eval_dir}/eval.txt

python utils/eval_metrics.py --eval_dir ${output_eval_dir} 2>&1 | tee ${output_eval_dir}/metrics.txt
