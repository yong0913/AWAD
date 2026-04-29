#!/bin/bash
device=0

depth=(9)
n_ctx=(12)
t_n_ctx=(4)

# ==========================================
# 1. 在 VisA 数据集上训练 (用于测试 MVTec)
# ==========================================
echo "🚀 开始在 VisA 数据集上训练 (带 正交正则化 + AAG-Pooling)..."
for i in "${!depth[@]}";do
    for j in "${!n_ctx[@]}";do
        # 【修改点】：加上了 _ortho_aag 后缀
        base_dir=${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_multiscale_visa_ortho_aag_t=0.13
        save_dir=./checkpoints/${base_dir}/
        mkdir -p ${save_dir}  # 确保文件夹存在
        LOG=${save_dir}"train.log"
        echo "日志将保存在: ${LOG}"
        
        CUDA_VISIBLE_DEVICES=${device} python train.py \
        --dataset visa \
        --train_data_path /root/autodl-tmp/AnomalyCLIP-main/dataset/visa \
        --save_path ${save_dir} \
        --features_list 6 12 18 24 \
        --image_size 518 \
        --batch_size 8 \
        --print_freq 1 \
        --epoch 15 \
        --save_freq 1 \
        --depth ${depth[i]} \
        --n_ctx ${n_ctx[j]} \
        --t_n_ctx ${t_n_ctx[0]} | tee ${LOG}  
    done
done

# ==========================================
# 2. 在 MVTec 数据集上训练 (用于测试 VisA)
# ==========================================
echo "🚀 开始在 MVTec 数据集上训练 (带 正交正则化 + AAG-Pooling)..."
for i in "${!depth[@]}";do
    for j in "${!n_ctx[@]}";do
        # 【修改点】：加上了 _ortho_aag 后缀
        base_dir=${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_multiscale_mvtec_ortho_aag_t=0.13
        save_dir=./checkpoints/${base_dir}/
        mkdir -p ${save_dir}
        LOG=${save_dir}"train.log"
        echo "日志将保存在: ${LOG}"

        CUDA_VISIBLE_DEVICES=${device} python train.py \
        --dataset mvtec \
        --train_data_path /root/autodl-tmp/AnomalyCLIP-main/dataset/mvtec \
        --save_path ${save_dir} \
        --features_list 6 12 18 24 \
        --image_size 518 \
        --batch_size 8 \
        --print_freq 1 \
        --epoch 15 \
        --save_freq 1 \
        --depth ${depth[i]} \
        --n_ctx ${n_ctx[j]} \
        --t_n_ctx ${t_n_ctx[0]} | tee ${LOG}
    done
done

echo "🎉 所有训练任务已完成！"