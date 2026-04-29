#!/bin/bash
device=0
depth=(9)
n_ctx=(12)
t_n_ctx=(4)

# ==========================================================
# 实验 1: 测试 MVTec 数据集 (🚨 使用 VisA 训练的权重)
# ==========================================================
for i in "${!depth[@]}";do
    for j in "${!n_ctx[@]}";do
        # 【修改点 1】：读取 VisA 带有 _ortho_aag 的权重文件夹
        weight_dir=${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_multiscale_visa_ortho_aag_t=0.13
        
        # 【修改点 2】：结果保存到带有 _ortho_aag 的新名称下，防止覆盖之前的结果
        save_result_dir=${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_multiscale_visa_to_mvtec_ortho_aag_t=0.13
        
        CUDA_VISIBLE_DEVICES=${device} python test.py --dataset mvtec \
        --data_path /root/autodl-tmp/AnomalyCLIP-main/dataset/mvtec \
        --save_path ./results/${save_result_dir}/zero_shot \
        --checkpoint_path ./checkpoints/${weight_dir}/epoch_15.pth \
        --features_list 6 12 18 24 --image_size 518 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]}
    wait
    done
done

# ==========================================================
# 实验 2: 测试 VisA 数据集 (🚨 使用 MVTec 训练的权重)
# ==========================================================
for i in "${!depth[@]}";do
    for j in "${!n_ctx[@]}";do
        # 【修改点 3】：读取 MVTec 带有 _ortho_aag 的权重文件夹
        weight_dir=${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_multiscale_mvtec_ortho_aag_t=0.13
        
        # 【修改点 4】：结果保存到带有 _ortho_aag 的新名称下
        save_result_dir=${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_multiscale_mvtec_to_visa_ortho_aag_t=0.13
        
        CUDA_VISIBLE_DEVICES=${device} python test.py --dataset visa \
        --data_path /root/autodl-tmp/AnomalyCLIP-main/dataset/visa \
        --save_path ./results/${save_result_dir}/zero_shot \
        --checkpoint_path ./checkpoints/${weight_dir}/epoch_15.pth \
        --features_list 6 12 18 24 --image_size 518 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]}
    wait
    done
done