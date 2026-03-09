IDX=0

IFS=',' read -r -a array <<< "$IDX"
len_node=${#array[@]}

export PYTHONPATH=$PYTHONPATH:./
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

output_dir=./checkpoints/ckpts/RAF-DB

if [ -d ${output_dir} ];then
    echo "dir already exists"
else
    mkdir ${output_dir}
fi

if [ -d ${output_dir}/src ];then
    echo "src dir already exists"
else
    echo "save codes to src"
    mkdir ${output_dir}/src
    cp -r datasets ${output_dir}/src
    cp -r models ${output_dir}/src
    cp -r utils ${output_dir}/src
    cp -r scripts ${output_dir}/src
fi

CUDA_VISIBLE_DEVICES=$IDX OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=$len_node --master_port=25003 \
    utils/trainface.py \
    --model_name_or_path ./checkpoints/model_weights/vicuna-7b-v1.5 \
    --llama_path ./checkpoints/model_weights/vicuna-7b-v1.5 \
    --data_path data_list/train/raf-db-des.txt \
    --dino_path ./checkpoints/model_weights/dinov2_vitl14_pretrain.pth \
    --conv_format face_task \
    --question_index -1 \
    --data_augmentation True \
    --tune_mm_mlp_adapter True \
    --freeze_llm False \
    --lora_llm_enable True \
    --freeze_vit False \
    --lora_vision_enable True \
    --bf16 True \
    --output_dir ${output_dir} \
    --num_train_epochs 10 \
    --per_device_train_batch_size 80 \
    --per_device_eval_batch_size 80 \
    --gradient_accumulation_steps 2 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 2 \
    --learning_rate 5e-4 \
    --weight_decay 0.05 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 50 \
    --dataloader_num_workers 8 \
    --tf32 True \
    --model_max_length 512 \
    --gradient_checkpointing True \
    --report_to tensorboard \
    2>&1 | tee ${output_dir}/log.txt


if [ -f "${output_dir}/config.json" ]; then
    output_eval_dir=${output_dir}/eval
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
else
    echo "skip eval: ${output_dir}/config.json not found (training likely failed)"
fi

if [ -f "${output_dir}/config.json" ]; then
    output_eval_dir=${output_dir}/eval-des
    if [ -d ${output_eval_dir} ];then
        echo "dir already exists"
    else
        mkdir ${output_eval_dir}
    fi

    CUDA_VISIBLE_DEVICES=$IDX OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=$len_node --master_port=25003 \
        utils/validfaceEMO-des.py \
        --model-name ${output_dir} \
        --question-file data_list/test/raf-db-91.03-des.txt \
        --output-dir ${output_eval_dir} \
        --conv-format facetask  2>&1 | tee ${output_eval_dir}/eval.txt
else
    echo "skip eval-des: ${output_dir}/config.json not found (training likely failed)"
fi
