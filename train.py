import os
import math
import argparse
import torch
import torch.optim.lr_scheduler as lr_scheduler
from fsspec.registry import default
from torch.utils.tensorboard import SummaryWriter
import json
from MyDataset import MyDataset
from VideoTransform import get_transforms_from_args, VideoTransform
from model_selector import select_model
from utils_data import read_split_data, train_one_epoch, evaluate, summarize_kfold_results, get_dataset_info


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model_dir = f'./weights/'
    if os.path.exists(model_dir) is False:
        os.makedirs(model_dir)

    tb_writer = SummaryWriter(f'./logs/{args.model_name}_{args.aug}_{args.train_dataset}_{args.scene}_{args.seed}')
    train_fs, train_video = get_dataset_info(args.train_dataset)
    val_fs, val_video = get_dataset_info(args.val_dataset)


    all_folds = read_split_data(dataset_name=args.train_dataset, Train_len=args.train_len, seed=args.seed, scene=args.scene)
    fold_metrics = []

    for fold_idx, (train_list, val_list) in enumerate(all_folds):
        print(f"\n========== Fold {fold_idx + 1}/5 ==========")

        # 定义数据增强管道
        #  "augment_gaussian_noise",      # G
        #  "augment_time_reversal",       # R
        #  "augment_horizontal_flip",     # H
        #  "augment_illumination_noise",  # I
        #  "augment_random_resized_crop", # C
        #  "augmentation_time_adapt"      # T
        selected_transforms = get_transforms_from_args(args.aug)
        data_transform = VideoTransform(selected_transforms)
        print(f'---------使用{selected_transforms}数据增强-------------')
    
        # 实例化训练数据集
        train_data = MyDataset(data_list=train_list,
                                  T=args.train_len,
                                  transform=data_transform,
                                  method='train',
                                  fs=train_fs)
    
        # # 实例化验证数据集
        val_data = MyDataset(data_list=val_list,
                                T=args.val_len,
                                transform=None,
                                method='val',
                                fs=val_fs)
    
    
    
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
    
        create_model = select_model(args.model_name, len=args.train_len)
        model = create_model.to(device)

    
        pg = [p for p in model.parameters() if p.requires_grad]
        optimizer = {
            "adamw": torch.optim.AdamW(pg, lr=args.lr),
            "adam": torch.optim.Adam(pg, lr=args.lr),
            "sgd": torch.optim.SGD(pg, lr=args.lr, momentum=0.9)
        }[args.optimizer]
    
        # Scheduler https://arxiv.org/pdf/1812.01187.pdf
        lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    
    
        best_score = float('inf')  # 设为一个很大的初始值
        best_epoch = 1
        w1, w2= 0.4, 0.6
    
        for epoch in range(1,args.epochs+1):
            # train
            train_loss,  train_mae, train_snr = train_one_epoch(model=model,
                                                    optimizer=optimizer,
                                                    data_loader=train_loader,
                                                    device=device,
                                                    epoch=epoch,
                                                    fs=train_fs,
                                                    loss_name=args.loss_name)
    
            scheduler.step()

            # 记录评价指标
            tags = ["train_loss", "train_mae", "train_snr", "learning_rate"]
            tb_writer.add_scalar(tags[0], train_loss, epoch)
            tb_writer.add_scalar(tags[1], train_mae, epoch)
            tb_writer.add_scalar(tags[2], train_snr, epoch)
            tb_writer.add_scalar(tags[3], optimizer.param_groups[0]["lr"], epoch)

            if epoch == args.epochs:
                checkpoint_path = os.path.join(model_dir, f"{args.model_name}_{args.train_dataset}_fold{fold_idx + 1}_{args.seed}_{args.aug}_last.pth")
                torch.save(model.state_dict(), checkpoint_path)
                print(f"the last epoch {epoch} 模型保存到 {checkpoint_path}")
    
            score = train_loss *w1 + train_mae * w2
    
            if score < best_score:
                best_score = score  # 更新最优 Loss
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(model_dir, f"{args.model_name}_{args.train_dataset}_fold{fold_idx + 1}_{args.seed}_{args.aug}_best.pth"))
            print(f"-----------------Best epoch: {best_epoch} | 当前是epcoh: {epoch} , loss 为:{train_loss}, mae为:{train_mae}, snr为:{train_snr}, learning rate: {optimizer.param_groups[0]['lr']}---------------")


        # validate
        model_weight_path = f'./weights/{args.model_name}_{args.train_dataset}_fold{fold_idx + 1}_{args.seed}_{args.aug}'
        if args.weight_path == 'best':
            weight_path = model_weight_path + '_best.pth'
        else:
            weight_path = model_weight_path + '_last.pth'
        model.load_state_dict(torch.load(weight_path, map_location=device))

        metrics_dict = evaluate(model=model,
                                data_loader=val_loader,
                                device=device,
                                foldidx=fold_idx,
                                args=args,
                                fs=val_fs)

        fold_metrics.append({
            "fold": fold_idx + 1,
            "metrics": metrics_dict})

    # 计算5折平均指标
    final_results = summarize_kfold_results(fold_metrics)

    # 准备保存的信息
    val_results = {
        "model_name": args.model_name,
        "seed": args.seed,
        "train_dataset": args.train_dataset,
        "val_dataset": args.val_dataset,
        "val_len": args.val_len * val_fs,
        "augment": args.aug,
        "weights_path": f'./weights/{args.model_name}_{args.seed}_{args.train_dataset}_{args.weight_path}.pth',
        "visualization_path": args.plot_path,
        "val_num_samples": len(val_loader.dataset),
        "fold_metrics": fold_metrics,
        "final_mean_metrics": final_results
    }

    # 保存到JSON
    save_dir = "./save"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{args.model_name}_{args.seed}_{args.aug}_{args.train_dataset}_{args.val_dataset}_{args.val_len * val_fs}.json")

    with open(save_path, "w") as f:
        json.dump(val_results, f, indent=4)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-dataset', type=str, default='UBFCrPPG', help='train dataset name')
    parser.add_argument('--val-dataset', type=str, default='UBFCrPPG', help='val dataset name')
    parser.add_argument('--train-len', type=int, default=160, help='train Length (frames)')
    parser.add_argument('--val-len', type=int, default=10, help='val Length （s）')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--scene', type=str, default='raw', help='the different scene in dataset')
    parser.add_argument('--nw', type=int, default=0, help='num_workers')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--optimizer', type=str, default="adamw")
    parser.add_argument('--model-name', type=str, default='RhythmFormer', help='create model name')
    parser.add_argument('--weight-path', type=str, default='best', help='best: load the best model weights, last:load the last weights')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--aug', type=str, default='TH',help='record augment method')
    parser.add_argument('--seed', type=int,  default=42, help='random seed')
    parser.add_argument('--loss-name', type=str, default= 'NegPearson' ,help='loss name')
    parser.add_argument('--plot', type=str, default='blandaltman', help='wave: only plot wave fig., blandaltman: only plot blandaltman fig., both: plot two fig.')
    parser.add_argument('--plot-path', type=str, default="result/Plots", help='the path of fig save!')
    parser.add_argument('--hr-method', type=str, default='FFT', help='=calculate hr use FFT or Peak')

    opt = parser.parse_args()

    main(opt)
