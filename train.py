import os
import math
import argparse
import torch
import yaml
import gc
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import json
from MyDataset import MyDataset
from VideoTransform import get_transforms_from_args, VideoTransform
from model_selector import select_model
from utils_data import read_split_data, train_one_epoch, evaluate, summarize_kfold_results, get_dataset_info


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # Create folder for saving final results
    save_dir = "./result/save"
    os.makedirs(save_dir, exist_ok=True)

    # Create folder for saving model weights
    model_dir = f'./weights/'
    os.makedirs(model_dir, exist_ok=True)

    plot_dir = './result/plots'
    os.makedirs(plot_dir, exist_ok=True)


    # Get frame rates and video counts for train and validation datasets
    train_fs, train_video = get_dataset_info(args.train_dataset)
    val_fs, val_video = get_dataset_info(args.val_dataset)

    # Check if cross-dataset or cross-scene validation is needed
    if args.train_dataset != args.val_dataset or args.scene[0] != args.scene[1]:
        print('Only test!')
        print(f'{args.model_name} train on {args.train_dataset} scene {args.scene[0]}, val on {args.val_dataset} scene {args.scene[1]}')

        # Load test set for cross-scene/domain evaluation
        val_list =read_split_data(dataset_name=args.val_dataset, dataset_root= args.dataset_root,Train_len=args.val_len, seed=args.seed, scene=args.scene[1], tag='cross')
        val_data = MyDataset(data_list=val_list,
                             T=args.val_len,
                             transform=None,
                             method='val',
                             fs=val_fs)

        val_loader = torch.utils.data.DataLoader(val_data,
                                                 batch_size=1,
                                                 shuffle=False,
                                                 pin_memory=True,
                                                 num_workers=args.nw,
                                                 collate_fn=val_data.collate_fn)

        create_model = select_model(args.model_name, len=args.train_len * val_fs)
        model = create_model.to(device)

        fold_metrics = []  # Store metrics for each fold
        for fold_idx in range(5):
            print(f"\n========== Fold {fold_idx + 1}/5 ==========")

            # Load trained model weights for current fold
            model_weight_path = f'./weights/{args.model_name}_{args.train_dataset}_scene{args.scene[0]}_fold{fold_idx + 1}_{args.seed}_{args.aug}'
            weight_path = model_weight_path + ('_best.pth' if args.weight_path == 'best' else '_last.pth')
            model.load_state_dict(torch.load(weight_path, map_location=device))

            # Evaluate model
            metrics_dict = evaluate(model=model,
                                    data_loader=val_loader,
                                    device=device,
                                    foldidx=fold_idx,
                                    args=args,
                                    fs=val_fs,
                                    plot_path=plot_dir)

            fold_metrics.append({
                "fold": fold_idx + 1,
                "metrics": metrics_dict})

            # Save evaluation results
            save_path = os.path.join(save_dir,
                                     f"{args.model_name}_{args.train_dataset}_scene{args.scene[0]}_{args.val_dataset}_scene{args.scene[1]}_{args.seed}_{args.aug}_{args.val_len * val_fs}.json")
            # 检查是否已有文件，读取已有内容
            if os.path.exists(save_path):
                with open(save_path, "r") as f:
                    saved_data = json.load(f)
            else:
                saved_data = {
                    "model_name": args.model_name,
                    "seed": args.seed,
                    "train_dataset": args.train_dataset,
                    "train_scene":args.scene[0],
                    "val_dataset": args.val_dataset,
                    "val_scene": args.scene[1],
                    "val_len": args.val_len * val_fs,
                    "augment": args.aug,
                    "val_num_samples": len(val_loader.dataset),
                    "fold_metrics": []
                }

            # 追加当前 fold
            saved_data["fold_metrics"].append({
                "fold": fold_idx + 1,
                "metrics": metrics_dict
            })

            # 保存回文件
            with open(save_path, "w") as f:
                json.dump(saved_data, f, indent=4)

    else:
        print('Train and test!')
        print(f'{args.model_name} train on {args.train_dataset} scene {args.scene[0]}, val on {args.val_dataset} scene {args.scene[1]}')

        tb_writer = SummaryWriter(f'./logs/{args.model_name}_{args.aug}_{args.train_dataset}_{args.scene[0]}_{args.seed}')

        all_folds = read_split_data(dataset_name=args.train_dataset, dataset_root= args.dataset_root, Train_len=args.train_len, seed=args.seed, scene=args.scene[0], tag='intra')

        fold_metrics = []  # Store metrics for each fold
        for fold_idx, (train_list, val_list) in enumerate(all_folds):
            print(f"\n========== Fold {fold_idx + 1}/5 ==========")

            # Define data augmentation
            #  "augment_gaussian_noise",      # G
            #  "augment_time_reversal",       # R
            #  "augment_horizontal_flip",     # H
            #  "augment_illumination_noise",  # I
            #  "augment_random_resized_crop", # C
            #  "augmentation_time_adapt"      # T
            selected_transforms = get_transforms_from_args(args.aug)
            data_transform = VideoTransform(selected_transforms)
            print(f'Using {selected_transforms} data augmentation')

            # Create training dataset
            train_data = MyDataset(data_list=train_list,
                                      T=args.train_len,
                                      transform=data_transform,
                                      method='train',
                                      fs=train_fs)

            val_data = MyDataset(data_list=val_list,
                                    T=args.val_len,
                                    transform=None,
                                    method='val',
                                    fs=val_fs)

            # Create data loaders
            train_loader = torch.utils.data.DataLoader(train_data,
                                                       batch_size=args.batch_size,
                                                       shuffle=True,
                                                       pin_memory=True,
                                                       num_workers=args.nw,
                                                       collate_fn=train_data.collate_fn)

            val_loader = torch.utils.data.DataLoader(val_data,
                                                     batch_size=1,
                                                     shuffle=False,
                                                     pin_memory=True,
                                                     num_workers=args.nw,
                                                     collate_fn=val_data.collate_fn)

            # Instantiate model
            create_model = select_model(args.model_name, len=args.train_len)
            model = create_model.to(device)

            # Set optimizer
            pg = [p for p in model.parameters() if p.requires_grad]
            optimizer = {
                "adamw": torch.optim.AdamW(pg, lr=args.lr),
                "adam": torch.optim.Adam(pg, lr=args.lr),
                "sgd": torch.optim.SGD(pg, lr=args.lr, momentum=0.9)
            }[args.optimizer]

            # Define cosine learning rate scheduler
            lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)


            best_score = float('inf')  # Initialize best score for early stopping
            best_epoch = 1
            w1, w2= 0.4, 0.6  # Weights for score combination

            for epoch in range(1,args.epochs+1):
                # train
                train_loss,  train_mae, train_snr = train_one_epoch(model=model,
                                                        optimizer=optimizer,
                                                        data_loader=train_loader,
                                                        device=device,
                                                        epoch=epoch,
                                                        fs=train_fs,
                                                        loss_name=args.loss_name)

                scheduler.step()  # Update learning rate

                # Log training metrics
                tags = ["train_loss", "train_mae", "train_snr", "learning_rate"]
                tb_writer.add_scalar(tags[0], train_loss, epoch)
                tb_writer.add_scalar(tags[1], train_mae, epoch)
                tb_writer.add_scalar(tags[2], train_snr, epoch)
                tb_writer.add_scalar(tags[3], optimizer.param_groups[0]["lr"], epoch)

                # Save model at last epoch
                if epoch == args.epochs:
                    checkpoint_path = os.path.join(model_dir, f"{args.model_name}_{args.train_dataset}_scene{args.scene[0]}_fold{fold_idx + 1}_{args.seed}_{args.aug}_last.pth")
                    torch.save(model.state_dict(), checkpoint_path)
                    print(f"the last epoch {epoch} 模型保存到 {checkpoint_path}")

                score = train_loss *w1 + train_mae * w2
                if score < best_score:
                    best_score = score  # 更新最优 Loss
                    best_epoch = epoch
                    torch.save(model.state_dict(), os.path.join(model_dir, f"{args.model_name}_{args.train_dataset}_scene{args.scene[0]}_fold{fold_idx + 1}_{args.seed}_{args.aug}_best.pth"))
                print(f"Best epoch: {best_epoch} | 当前是epcoh: {epoch} , loss 为:{train_loss}, mae为:{train_mae}, snr为:{train_snr}, learning rate: {optimizer.param_groups[0]['lr']}")


            del model
            torch.cuda.empty_cache()
            gc.collect()
            # validate
            model_weight_path = f'./weights/{args.model_name}_{args.train_dataset}_scene{args.scene[0]}_fold{fold_idx + 1}_{args.seed}_{args.aug}'
            if args.weight_path == 'best':
                weight_path = model_weight_path + '_best.pth'
            else:
                weight_path = model_weight_path + '_last.pth'

            create_model = select_model(args.model_name, args.val_len * val_fs)
            model = create_model.to(device)
            model.load_state_dict(torch.load(weight_path, map_location=device))

            metrics_dict = evaluate(model=model,
                                    data_loader=val_loader,
                                    device=device,
                                    foldidx=fold_idx,
                                    args=args,
                                    fs=val_fs,
                                    plot_path=plot_dir)

            del model
            torch.cuda.empty_cache()
            gc.collect()
            fold_metrics.append({
                "fold": fold_idx + 1,
                "metrics": metrics_dict})

            save_path = os.path.join(save_dir, f"{args.model_name}_{args.train_dataset}_scene{args.scene[0]}_{args.val_dataset}_scene{args.scene[1]}_{args.seed}_{args.aug}_{args.val_len * val_fs}.json")

            if os.path.exists(save_path):
                with open(save_path, "r") as f:
                    saved_data = json.load(f)
            else:
                saved_data = {
                    "model_name": args.model_name,
                    "seed": args.seed,
                    "train_dataset": args.train_dataset,
                    "train_scene":args.scene[0],
                    "val_dataset": args.val_dataset,
                    "val_scene": args.scene[1],
                    "val_len": args.val_len * val_fs,
                    "augment": args.aug,
                    "val_num_samples": len(val_loader.dataset),
                    "fold_metrics": []
                }

            saved_data["fold_metrics"].append({
                "fold": fold_idx + 1,
                "metrics": metrics_dict
            })

            with open(save_path, "w") as f:
                json.dump(saved_data, f, indent=4)


    # Compute final average metrics over all folds
    final_results = summarize_kfold_results(fold_metrics)
    with open(save_path, "r") as f:
        saved_data = json.load(f)
    saved_data["final_mean_metrics"] = final_results

    with open(save_path, "w") as f:
        json.dump(saved_data, f, indent=4)


def parse_args():
    # 第一步：定义完整参数结构（含默认值）
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='Path to config file (YAML)')
    parser.add_argument('--train_dataset', type=str, default='DLCN')
    parser.add_argument('--val_dataset', type=str, default='DLCN')
    parser.add_argument('--train_len', type=int, default=160)
    parser.add_argument('--val_len', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--scene', nargs='+', type=str, default=['FIFP', 'FIVP'])
    parser.add_argument('--nw', type=int, default=0)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--optimizer', type=str, default="adamw")
    parser.add_argument('--model_name', type=str, default='RhythmFormer')
    parser.add_argument('--weight_path', type=str, default='best')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--aug', type=str, default='TH')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--loss_name', type=str, default='NegPearson')
    parser.add_argument('--plot', type=str, default='blandaltman')
    parser.add_argument('--plot_path', type=str, default="result/plots")
    parser.add_argument('--dataset_root', type=str, default="D:\\Dataset")
    parser.add_argument('--hr_method', type=str, default='FFT')

    # 第二步：先解析出 config 路径
    config_args, remaining_args = parser.parse_known_args()

    # 第三步：从 YAML 中加载默认值
    if config_args.config:
        with open(config_args.config, 'r', encoding='utf-8') as f:
            yaml_cfg = yaml.safe_load(f)

        # 将 YAML 中的值作为默认值设置
        for key, value in yaml_cfg.items():
            arg_name = f'--{key.replace("_", "-")}' if '_' in key else f'--{key}'
            if any(a.dest == key for a in parser._actions):
                parser.set_defaults(**{key: value})
            else:
                print(f"⚠️  YAML 配置中包含未注册参数: {key}")

    # 第四步：最终解析参数
    args = parser.parse_args(remaining_args)
    return args


if __name__ == '__main__':

    opt = parse_args()
    main(opt)