import sys
import torch
from tqdm import tqdm
import os
import json
import random
import numpy as np
from collections import defaultdict
from sklearn.model_selection import KFold
from model_selector import select_loss
from post_process import calculate_metric_batch_video, calculate_metric_per_video ,calculate_metrics
from visual import plot_wave_psd, plot_blandaltman

def read_split_data(dataset_name: str = "UBFCrPPG", Train_len: int = 160, seed: int = 42, scene:str = 'Raw', tag: str = 'intra'):
    random.seed(seed)
    np.random.seed(seed)

    data_root = f"D:\\Dataset\\{dataset_name}"
    assert os.path.exists(data_root), f"dataset root: {data_root} does not exist."

    # 加载并排序文件路径
    if dataset_name == 'UBFCrPPG':
        files = sorted([f for f in os.listdir(data_root) if f.endswith(".h5")],
                       key=lambda x: int(x.split(".")[0]))
    elif dataset_name == 'COHFACE':
        files = sorted([f for f in os.listdir(data_root) if f.endswith(".h5")],
                       key=lambda x: (int(x.split("-")[0]), int(x.split("-")[1].split(".")[0])))
    elif dataset_name == 'MMPD':
        files = sorted([f for f in os.listdir(data_root) if f.endswith(".h5")],
                       key=lambda x: int(x.split("_")[0][1:]))
    elif dataset_name == 'PURE':
        files = sorted([f for f in os.listdir(data_root) if f.endswith(".h5")],
                       key=lambda x: (int(x.split("-")[0]), int(x.split("-")[1].split(".")[0])))

    elif dataset_name == 'DLCN':
        # 映射模4结果到场景名
        scene_mapping = {
            1: 'FIFP',
            2: 'VIFP',
            3: 'FIVP',
            0: 'VIVP'  # 注意：idx % 4 == 0 表示原始 idx 是 4 或 8
        }
        # 用于存储每个场景对应的文件列表
        scene_data = defaultdict(list)
        rest_files = []
        exercise_files = []

        for file in os.listdir(data_root):
            if file.endswith('.h5') and file.startswith('P'):
                name = file.split('.')[0]  # eg. 'P1_3'
                _, idx = name.split('_')
                idx = int(idx)

                scene_id = idx % 4
                scene_name = scene_mapping[scene_id]
                scene_data[scene_name].append(file)

                # 判断属于哪种状态
                if 1 <= idx <= 4:
                    rest_files.append(file)
                elif 5 <= idx <= 8:
                    exercise_files.append(file)

        if scene == 'R':
            files = rest_files
            print(f"静止状态样本个数: {len(files)}")

        elif scene == 'E':
            files = exercise_files
            print(f"\n运动状态样本个数: {len(files)}")

        elif scene == 'FIFP':
            files = scene_data[scene]
            print(f"\n光强固定且位置固定: {len(files)}")
        elif scene == 'VIFP':
            files = scene_data[scene]
            print(f"\n光强变化且位置固定: {len(files)}")
        elif scene == 'FIVP':
            files = scene_data[scene]
            print(f"\n光强固定且位置变化: {len(files)}")
        elif scene == 'VIVP':
            files = scene_data[scene]
            print(f"\n光强变化且位置变化: {len(files)}")
        elif scene == 'Raw':
            files = [f for f in os.listdir(data_root) if f.endswith(".h5")]
            print(f"\n所有数据: {len(files)}")
        else:
            raise ValueError(f"Unknown scene: {scene}")

        files = sorted(files,
            key=lambda x: (int(x.split("_")[0][1:]), int(x.split("_")[1].split(".")[0]))
        )

    file_paths = [os.path.join(data_root, f) for f in files]
    print(f'all of num: {len(file_paths)}')

    if tag == 'cross':
        json_path = f"./dataconfig/{tag}_{dataset_name}_scene{scene}_seed{seed}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "val_len": len(file_paths),
                "val": file_paths
            }, f, indent=4)
        print(f'val num: {len(file_paths)}')
        return file_paths
    elif tag == 'intra':

        # 设置fps和video_len
        fs, video_len = get_dataset_info(dataset_name)

        num_repeat = np.round(video_len / (Train_len / fs)).astype(int)

        # 5折划分
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        splits = list(kf.split(file_paths))

        all_folds = []
        os.makedirs("./dataconfig/", exist_ok=True)

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            train_set = [file_paths[i] for i in train_idx]
            val_set = [file_paths[i] for i in val_idx]

            train_set_expanded = train_set * num_repeat

            all_folds.append((train_set_expanded, val_set))

            # 保存配置文件
            json_path = f"./dataconfig/{tag}_{dataset_name}_5fold_fold{fold_idx}_scene{scene}_seed{seed}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({
                    "fold_index": fold_idx,
                    "train_len": len(train_set),
                    "train": train_set,
                    "val_len": len(val_set),
                    "val": val_set
                }, f, indent=4)
        print(f'train num: {len(train_set)}, val num: {len(val_set)}')
        return all_folds
    else:
        raise ValueError(f"tag only intra or cross, now tag:{tag}")


def get_dataset_info(dataset_name):
    """
    根据数据集名称返回该数据集的采样率 fs 和视频长度 video_len（单位为秒）

    参数:
        dataset_name (str): 数据集名称，比如 'PURE', 'UBFC-rPPG', 'DCLN', 等等

    返回:
        fs (int or float): 帧率 (frames per second)
        video_len (int or float): 视频总时长（单位为秒）
    """

    dataset_info = {
        "PURE": {"fs": 30, "video_len": 60},
        "UBFCrPPG": {"fs": 30, "video_len": 60},
        "UBFCPhys": {"fs": 35, "video_len": 180},
        "COHFACE": {"fs": 20, "video_len": 60},
        "DLCN": {"fs": 30, "video_len": 60},  # 你自定义的数据集
        "MMSE-HR": {"fs": 25, "video_len": 60},
        "VIPL-HR": {"fs": 30, "video_len": 30},
        "MMPD": {"fs": 30, "video_len": 60},
        # 根据需要补充更多数据集
    }

    if dataset_name not in dataset_info:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    return dataset_info[dataset_name]["fs"], dataset_info[dataset_name]["video_len"]



def train_one_epoch(model, optimizer, data_loader, device, epoch, fs, loss_name):
    model.train()

    loss_function = select_loss(loss_name, fs)
    accu_loss = torch.zeros(1).to(device)  # 累计损失


    accu_mae = np.zeros(1)   # 累计预测mae
    accu_SNR = np.zeros(1)  # 累计预测snr
    # accu_MACC= np.zeros(1)  # 累计预测MACC


    optimizer.zero_grad()

    sample_num = 0
    print('train_datalen:', len(data_loader))
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]
        pred = model(images.to(device))

        pred_np = pred.cpu().detach().numpy()
        label_np = labels.cpu().detach().numpy()

        hr_pred, hr_label, SNR,  MACC = calculate_metric_batch_video(pred_np, label_np, fs)
        accu_mae += np.abs(hr_pred - hr_label).sum()
        accu_SNR += SNR.sum()
        # accu_MACC += MACC.sum()


        loss = loss_function(pred, labels.to(device), epoch, fs)
        loss.backward()

        # 打印反向传播的参数
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         print(f"{name}: grad mean={param.grad.mean().item()}, grad max={param.grad.max().item()}")
        #     else:
        #         print(f"{name}: grad is None!")

        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss_all: {:.3f}, hr_mae: {:.3f}, snr: {:.3f}".format(
                                                                        epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_mae.item() / sample_num,
                                                                               accu_SNR.item() / sample_num,
                                                                               )

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return (accu_loss.item() / (step + 1), accu_mae.item() / sample_num, accu_SNR.item() / sample_num)


@torch.no_grad()
def evaluate(model, data_loader, device, foldidx, args, fs):

    os.makedirs(args.plot_path, exist_ok=True)
    model.eval()

    hr_pred_all = []  # 累计预测mae
    hr_label_all = []
    snr_all = []  # 累计预测snr
    r_all = []

    data_loader = tqdm(data_loader, file=sys.stdout)
    val_len = args.val_len * fs
    for step, data in enumerate(data_loader):
        images, labels = data  # images:[B,C,T,H,W], T:[B,T]
        num_repeat = np.floor(images.shape[2] / val_len).astype(int)  # 计算 num_batches

        # pred_all = []
        # label_all = []

        for i in range(num_repeat):

            img_window = images[:, :, i * val_len:(i + 1) * val_len, :, :]
            label_window = labels[:, i * val_len:(i + 1) * val_len]

            pred = model(img_window.to(device))

            pred_np = pred.detach().cpu().numpy()
            label_np = label_window.detach().cpu().numpy()

            if args.plot in ["wave", "both"]:
                fig_name = f'wave_{args.model_name}_{args.train_dataset}_scene{args.scene[0]}_{args.val_dataset}_scene{args.scene[1]}_fold{foldidx}_seed{args.seed}_aug{args.aug}_{step}_{i}'
                fig_path = os.path.join(args.plot_path, fig_name)
                plot_wave_psd(pred_np, label_np, fs, fig_path)

            # pred_all.append(pred_np)
            # label_all.append(label_np)

            hr_pred, hr_label, snr, r = calculate_metric_per_video(pred_np, label_np, fs,
                                                                   hr_method=args.hr_method)
            # print("hr_pred:{}, hr_label:{}".format(hr_pred, hr_label))
            hr_pred_all.append(hr_pred)
            hr_label_all.append(hr_label)
            snr_all.append(snr)
            r_all.append(r)

        # pred_all = np.array(pred_all)
        # label_all = np.array(label_all)

        data_loader.desc = 'val on {}'.format(step)

    if args.plot in ["blandaltman", "both"]:
        fig_name = f'blandaltman_{args.model_name}_{args.train_dataset}_scene{args.scene[0]}_{args.val_dataset}_scene{args.scene[1]}_fold{foldidx}__seed{args.seed}_aug{args.aug}_{val_len}'
        fig_path = os.path.join(args.plot_path, fig_name)

        plot_blandaltman(hr_pred_all, hr_label_all, fig_path)

    metrics_dict = calculate_metrics(hr_pred_all, hr_label_all, snr_all, r_all)

    return metrics_dict

def summarize_kfold_results(fold_metrics):
    metric_names = ["MAE", "RMSE", "MAPE", "Pearson", "SNR", "MACC"]
    values_across_folds = {name: [] for name in metric_names}
    stds_across_folds = {name: [] for name in metric_names}

    for fold_result in fold_metrics:
        metrics = fold_result["metrics"]
        for metric in metric_names:
            values_across_folds[metric].append(metrics[metric]["value"])

    final_metrics = {}
    for metric in metric_names:
        value_list = values_across_folds[metric]
        std_list = stds_across_folds[metric]
        final_metrics[metric] = {
            "mean": np.mean(value_list),
            "std": np.std(value_list, ddof=1),         # 5个fold的 value 的 std
        }

    print("5-fold cross validation summary:")
    for metric, vals in final_metrics.items():
        print(f"{metric}: {vals['mean']:.3f}, +/- {vals['std']:.3f}")

    return final_metrics





if __name__ == '__main__':
    flods = read_split_data('DLCN',  seed=42, scene='Raw', tag='intra')


