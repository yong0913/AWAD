import DPDN_lib
import torch
import argparse
import torch.nn.functional as F
from prompt_ensemble import DPND_PromptLearner
from loss import FocalLoss, BinaryDiceLoss
from dataset import Dataset
from logger import get_logger
from tqdm import tqdm
import numpy as np
import os
import random
from utils import get_transform

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(args):
    logger = get_logger(args.save_path)
    preprocess, target_transform = get_transform(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    DPDN_parameters = {"Prompt_length": args.n_ctx, "learnabel_text_embedding_depth": args.depth, "learnabel_text_embedding_length": args.t_n_ctx}

    model, _ = DPDN_lib.load("ViT-L/14@336px", device=device, design_details = DPDN_parameters)
    model.eval()

    train_data = Dataset(root=args.train_data_path, transform=preprocess, target_transform=target_transform, dataset_name = args.dataset)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    num_layers = len(args.features_list) # 默认 4 层
    prompt_learner = DPDN_PromptLearner(model.to("cpu"), DPDN_parameters, num_layers=num_layers)
    prompt_learner.to(device)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer = 20)

    optimizer = torch.optim.Adam(list(prompt_learner.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))

    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    lam = 4
    alpha_ortho = 0.1 

    for epoch in tqdm(range(args.epoch)):
        model.eval()
        prompt_learner.train()
        loss_list = []
        image_loss_list = []

        for items in tqdm(train_dataloader):
            image = items['img'].to(device)
            label = items['anomaly']

            gt = items['img_mask'].squeeze().to(device)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0

            with torch.no_grad():
                image_features, patch_features = model.encode_image(image, args.features_list, DPAM_layer = 20)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    
            prompts, tokenized_prompts, compound_prompts_text = prompt_learner()
            
            # shape: [4, 2, 768]
            text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
            text_features = text_features.view(num_layers, 2, -1)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            loss_ortho = 0.0
            for i in range(num_layers):
                for j in range(i + 1, num_layers):
                    sim = F.cosine_similarity(text_features[i], text_features[j], dim=-1).mean()
                    loss_ortho += torch.abs(sim)
            global_text_features = text_features[-1]  # 获取最深层文本特征 [2, 768]

            deep_patch_features = patch_features[-1]  
            deep_patch_features = deep_patch_features / deep_patch_features.norm(dim=-1, keepdim=True)

            patch_sim = deep_patch_features @ global_text_features.T
            anomaly_scores = patch_sim[:, :, 1]  # 提取异常维度的得分

            temperature = 0.13  
            attention_weights = F.softmax(anomaly_scores / temperature, dim=-1)

            attention_weights = attention_weights.unsqueeze(-1)
            aag_global_feature = torch.sum(deep_patch_features * attention_weights, dim=1)  
            aag_global_feature = aag_global_feature / aag_global_feature.norm(dim=-1, keepdim=True)

            text_probs = aag_global_feature @ global_text_features.T

            if text_probs.dim() == 1:
                text_probs = text_probs.unsqueeze(0)
            text_probs = text_probs / 0.07

            image_loss = F.cross_entropy(text_probs, label.long().cuda())
            image_loss_list.append(image_loss.item())

            similarity_map_list = []
            for idx, patch_feature in enumerate(patch_features):
                if idx >= args.feature_map_layer[0]:
                    patch_feature = patch_feature / patch_feature.norm(dim = -1, keepdim = True)
                    layer_text_features = text_features[idx] 
                    similarity, _ = DPDN_lib.compute_similarity(patch_feature, layer_text_features)
                    similarity_map = DPDN.get_similarity_map(similarity[:, 1:, :], args.image_size).permute(0, 3, 1, 2)
                    similarity_map_list.append(similarity_map)

            loss = 0
           
            for i in range(len(similarity_map_list)):
                layer_loss = loss_focal(similarity_map_list[i], gt)
                layer_loss += loss_dice(similarity_map_list[i][:, 1, :, :], gt)
                layer_loss += loss_dice(similarity_map_list[i][:, 0, :, :], 1-gt)
                loss += layer_loss

            loss = lam * loss
            loss = loss + alpha_ortho * loss_ortho 

            optimizer.zero_grad()
            (loss + image_loss).backward()
            optimizer.step()
            loss_list.append(loss.item())
            
        if (epoch + 1) % args.print_freq == 0:
            logger.info('epoch [{}/{}], loss:{:.4f}, image_loss:{:.4f}'.format(epoch + 1, args.epoch, np.mean(loss_list), np.mean(image_loss_list)))

        if (epoch + 1) % args.save_freq == 0:
            ckp_path = os.path.join(args.save_path, 'epoch_' + str(epoch + 1) + '.pth')
            torch.save({"prompt_learner": prompt_learner.state_dict()}, ckp_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("DPDN", add_help=True)
    parser.add_argument("--train_data_path", type=str, default="./data/visa", help="train dataset path")
    parser.add_argument("--save_path", type=str, default='./checkpoint', help='path to save results')
    parser.add_argument("--dataset", type=str, default='mvtec', help="train dataset name")
    parser.add_argument("--depth", type=int, default=9, help="image size")
    parser.add_argument("--n_ctx", type=int, default=12, help="zero shot")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="zero shot")
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[0, 1, 2, 3], help="zero shot")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument("--epoch", type=int, default=15, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--print_freq", type=int, default=1, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=1, help="save frequency")
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    args = parser.parse_args()
    setup_seed(args.seed)
    train(args)