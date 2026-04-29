import AnomalyCLIP_lib
import torch
import argparse
import torch.nn.functional as F
from prompt_ensemble import AnomalyCLIP_PromptLearner
from dataset import Dataset
from logger import get_logger
from tqdm import tqdm
import os
import random
import numpy as np
from tabulate import tabulate
from utils import get_transform
from metrics import image_level_metrics, pixel_level_metrics
from scipy.ndimage import gaussian_filter

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def test(args):
    img_size = args.image_size
    features_list = args.features_list
    dataset_dir = args.data_path
    save_path = args.save_path
    dataset_name = args.dataset

    logger = get_logger(args.save_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    AnomalyCLIP_parameters = {"Prompt_length": args.n_ctx, "learnabel_text_embedding_depth": args.depth, "learnabel_text_embedding_length": args.t_n_ctx}
    
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details = AnomalyCLIP_parameters)
    model.eval()

    preprocess, target_transform = get_transform(args)
    test_data = Dataset(root=args.data_path, transform=preprocess, target_transform=target_transform, dataset_name = args.dataset)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    obj_list = test_data.obj_list

    results = {}
    metrics = {}
    for obj in obj_list:
        results[obj] = {}
        results[obj]['gt_sp'] = []
        results[obj]['pr_sp'] = []
        results[obj]['imgs_masks'] = []
        results[obj]['anomaly_maps'] = []
        metrics[obj] = {}

    num_layers = len(args.features_list)
    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), AnomalyCLIP_parameters, num_layers=num_layers)
    checkpoint = torch.load(args.checkpoint_path)
    prompt_learner.load_state_dict(checkpoint["prompt_learner"])
    prompt_learner.to(device)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer = 20)

    for idx, items in enumerate(tqdm(test_dataloader)):
        image = items['img'].to(device)
        cls_name = items['cls_name']
        cls_id = items['cls_id']
        
        gt_mask = items['img_mask']
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        results[cls_name[0]]['imgs_masks'].append(gt_mask.detach().cpu().numpy().astype(np.int8))
        results[cls_name[0]]['gt_sp'].extend(items['anomaly'].detach().cpu().numpy())

        with torch.no_grad():
            image_features, patch_features = model.encode_image(image, features_list, DPAM_layer = 20)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            prompts, tokenized_prompts, compound_prompts_text = prompt_learner()
            text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
            
            text_features = text_features.view(num_layers, 2, -1)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # =====================================================================
            # 🌟 创新点 4: AAG-Pooling (自适应异常引导池化) 测试端同步
            # =====================================================================
            global_text_features = text_features[-1]
            
            # 1. 获取最深层的 Patch 特征并归一化
            deep_patch_features = patch_features[-1]
            deep_patch_features = deep_patch_features / deep_patch_features.norm(dim=-1, keepdim=True)

            # 2. 计算每个 Patch 的异常得分
            patch_sim = deep_patch_features @ global_text_features.T
            anomaly_scores = patch_sim[:, :, 1]  # 提取异常维度得分

            # 3. 动态空间注意力权重
            temperature = 0.13
            attention_weights = F.softmax(anomaly_scores / temperature, dim=-1)

            # 4. 加权聚合出纯净的全新全局特征
            attention_weights = attention_weights.unsqueeze(-1)
            aag_global_feature = torch.sum(deep_patch_features * attention_weights, dim=1)
            aag_global_feature = aag_global_feature / aag_global_feature.norm(dim=-1, keepdim=True)

            # 5. 使用重构的全局特征计算最终 Image-level Anomaly Score
            text_probs = (aag_global_feature @ global_text_features.T / 0.07).softmax(-1)
            text_probs = text_probs[:, 1]
            # =====================================================================
            
            # === 像素级异常图 ===
            anomaly_map_list = []
            for idx, patch_feature in enumerate(patch_features):
                if idx >= args.feature_map_layer[0]:
                    patch_feature = patch_feature / patch_feature.norm(dim = -1, keepdim = True)
                    # 【核心消融点：用专属层级的 text_feature】
                    layer_text_features = text_features[idx]
                    similarity, _ = AnomalyCLIP_lib.compute_similarity(patch_feature, layer_text_features)
                    similarity_map = AnomalyCLIP_lib.get_similarity_map(similarity[:, 1:, :], args.image_size)
                    anomaly_map_layer = (similarity_map[...,1] + 1 - similarity_map[...,0])/2.0
                    anomaly_map_list.append(anomaly_map_layer)

            # 直接相加融合，没有加权
            anomaly_map = 0
            for i in range(len(anomaly_map_list)):
                anomaly_map += anomaly_map_list[i]

            results[cls_name[0]]['pr_sp'].extend(text_probs.detach().cpu().numpy())
            anomaly_map_np = np.stack([gaussian_filter(i, sigma = args.sigma) for i in anomaly_map.detach().cpu().numpy()], axis=0).astype(np.float32)
            results[cls_name[0]]['anomaly_maps'].append(anomaly_map_np)

    table_ls = []
    image_auroc_list = []
    image_ap_list = []
    pixel_auroc_list = []
    pixel_aupro_list = []
    
    for obj in obj_list:
        table = []
        table.append(obj)
        
        results[obj]['imgs_masks'] = np.concatenate(results[obj]['imgs_masks'], axis=0)
        results[obj]['anomaly_maps'] = np.concatenate(results[obj]['anomaly_maps'], axis=0)
        
        if args.metrics == 'image-level':
            image_auroc = image_level_metrics(results, obj, "image-auroc")
            image_ap = image_level_metrics(results, obj, "image-ap")
            table.append(str(np.round(image_auroc * 100, decimals=1)))
            table.append(str(np.round(image_ap * 100, decimals=1)))
            image_auroc_list.append(image_auroc)
            image_ap_list.append(image_ap) 
        elif args.metrics == 'pixel-level':
            pixel_auroc = pixel_level_metrics(results, obj, "pixel-auroc")
            pixel_aupro = pixel_level_metrics(results, obj, "pixel-aupro")
            table.append(str(np.round(pixel_auroc * 100, decimals=1)))
            table.append(str(np.round(pixel_aupro * 100, decimals=1)))
            pixel_auroc_list.append(pixel_auroc)
            pixel_aupro_list.append(pixel_aupro)
        elif args.metrics == 'image-pixel-level':
            image_auroc = image_level_metrics(results, obj, "image-auroc")
            image_ap = image_level_metrics(results, obj, "image-ap")
            pixel_auroc = pixel_level_metrics(results, obj, "pixel-auroc")
            pixel_aupro = pixel_level_metrics(results, obj, "pixel-aupro")
            table.append(str(np.round(pixel_auroc * 100, decimals=1)))
            table.append(str(np.round(pixel_aupro * 100, decimals=1)))
            table.append(str(np.round(image_auroc * 100, decimals=1)))
            table.append(str(np.round(image_ap * 100, decimals=1)))
            image_auroc_list.append(image_auroc)
            image_ap_list.append(image_ap) 
            pixel_auroc_list.append(pixel_auroc)
            pixel_aupro_list.append(pixel_aupro)
        table_ls.append(table)

    if args.metrics == 'image-pixel-level':
            table_ls.append(['mean', str(np.round(np.mean(pixel_auroc_list) * 100, decimals=1)),
                            str(np.round(np.mean(pixel_aupro_list) * 100, decimals=1)), 
                            str(np.round(np.mean(image_auroc_list) * 100, decimals=1)),
                            str(np.round(np.mean(image_ap_list) * 100, decimals=1))])
            results_formatted = tabulate(table_ls, headers=['objects', 'pixel_auroc', 'pixel_aupro', 'image_auroc', 'image_ap'], tablefmt="pipe")

    elif args.metrics == 'image-level':
        table_ls.append(['mean', str(np.round(np.mean(image_auroc_list) * 100, decimals=1)),
                        str(np.round(np.mean(image_ap_list) * 100, decimals=1))])
        results_formatted = tabulate(table_ls, headers=['objects', 'image_auroc', 'image_ap'], tablefmt="pipe")

    elif args.metrics == 'pixel-level':
        table_ls.append(['mean', str(np.round(np.mean(pixel_auroc_list) * 100, decimals=1)),
                        str(np.round(np.mean(pixel_aupro_list) * 100, decimals=1))])
        results_formatted = tabulate(table_ls, headers=['objects', 'pixel_auroc', 'pixel_aupro'], tablefmt="pipe")

    else:
        results_formatted = "Error: Unknown metrics"

    logger.info("\n%s", results_formatted)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("AnomalyCLIP", add_help=True)
    parser.add_argument("--data_path", type=str, default="./data/visa", help="path to test dataset")
    parser.add_argument("--save_path", type=str, default='./results/', help='path to save results')
    parser.add_argument("--checkpoint_path", type=str, default='./checkpoint/', help='path to checkpoint')
    parser.add_argument("--dataset", type=str, default='mvtec')
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--depth", type=int, default=9, help="image size")
    parser.add_argument("--n_ctx", type=int, default=12, help="zero shot")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="zero shot")
    parser.add_argument("--feature_map_layer", type=int,  nargs="+", default=[0, 1, 2, 3], help="zero shot")
    parser.add_argument("--metrics", type=str, default='image-pixel-level')
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    parser.add_argument("--sigma", type=int, default=4, help="zero shot")
    args = parser.parse_args()
    setup_seed(args.seed)
    test(args)