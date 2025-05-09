import sys
import torch
from tqdm import tqdm
import os
import json
import numpy as np
from collections import defaultdict
from sklearn.model_selection import KFold
from model_selector import select_loss
from post_process import calculate_metric_batch_video, calculate_metric_per_video ,calculate_metrics
from visual import plot_wave_psd, plot_blandaltman


def prepare_split_data(dataset_name: str, dataset_root='D:\\Dataset', train_len=160,
                       seed=42, scene='Raw', tag='intra'):
    """
    自动判断是否已划分数据集，若无则生成，再加载划分文件。
    """
    def check_split_exists():
        if tag == 'cross':
            json_path = f"./datasetinfo/{tag}_{dataset_name}_scene{scene}_seed{seed}.json"
            return os.path.exists(json_path)
        elif tag == 'intra':
            for fold_idx in range(5):
                json_path = f"./datasetinfo/{tag}_{dataset_name}_5fold_fold{fold_idx}_scene{scene}_seed{seed}.json"
                if not os.path.exists(json_path):
                    return False
            return True
        else:
            raise ValueError(f"tag must be 'intra' or 'cross', but got: {tag}")

    if not check_split_exists():
        print("[未发现数据集划分文件] 现在开始生成...")
        generate_split_data(dataset_name=dataset_name,
                            dataset_root=dataset_root,
                            seed=seed,
                            scene=scene,
                            tag=tag)
    else:
        print("[检测到已有数据集划分文件] 将直接加载")

    return load_split_data(dataset_name=dataset_name,
                           scene=scene,
                           seed=seed,
                           tag=tag,
                           train_len=train_len)


def generate_split_data(dataset_name: str, dataset_root='D:\\Dataset',
                        seed=42, scene='Raw', tag='intra'):
    """
    如果划分文件不存在，则进行划分并保存 JSON 文件。
    """
    data_root = os.path.join(dataset_root, dataset_name)
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
        elif scene == 'E':
            files = exercise_files
        elif scene == 'FIFP':
            files = scene_data[scene]
        elif scene == 'VIFP':
            files = scene_data[scene]
        elif scene == 'FIVP':
            files = scene_data[scene]
        elif scene == 'VIVP':
            files = scene_data[scene]
        elif scene == 'Raw':
            files = [f for f in os.listdir(data_root) if f.endswith(".h5")]
        else:
            raise ValueError(f"Unknown scene: {scene}")

        files = sorted(files,
            key=lambda x: (int(x.split("_")[0][1:]), int(x.split("_")[1].split(".")[0]))
        )

    file_paths = [os.path.join(data_root, f) for f in files]  #TODO debug
    print(f'所有数据数量: {len(file_paths)}')
    os.makedirs("./datasetinfo/", exist_ok=True)

    if tag == 'cross':
        json_path = f"./datasetinfo/{tag}_{dataset_name}_scene{scene}_seed{seed}.json"
        if not os.path.exists(json_path):
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({
                    "val_len": len(file_paths),
                    "val": file_paths
                }, f, indent=4)
            print(f"[生成] 跨数据集划分文件: {json_path}")

    elif tag == 'intra':
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        splits = list(kf.split(file_paths))

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            train_set = [file_paths[i] for i in train_idx]
            val_set = [file_paths[i] for i in val_idx]

            json_path = f"./datasetinfo/{tag}_{dataset_name}_5fold_fold{fold_idx}_scene{scene}_seed{seed}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({
                    "fold_index": fold_idx,
                    "train_len": len(train_set),
                    "train": train_set,
                    "val_len": len(val_set),
                    "val": val_set
                }, f, indent=4)
            print(f"[生成] 第{fold_idx}折划分文件: {json_path}")
    else:
        raise ValueError(f"tag must be 'intra' or 'cross', but got: {tag}")

def load_split_data(dataset_name: str, scene='Raw', seed=42, tag='intra', train_len=160):
    """
    加载已划分好的数据集文件。
    """
    if tag == 'cross':
        json_path = f"./datasetinfo/{tag}_{dataset_name}_scene{scene}_seed{seed}.json"
        assert os.path.exists(json_path), f"{json_path} not found!"
        with open(json_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return config["val"]

    if tag == 'intra':
        fs, video_len = get_dataset_info(dataset_name)
        num_repeat = np.round(video_len / (train_len / fs)).astype(int)
        all_folds=[]
        for fold_idx in range(5):
            json_path = f"./datasetinfo/{tag}_{dataset_name}_5fold_fold{fold_idx}_scene{scene}_seed{seed}.json"
            with open(json_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                train_set = config["train"]
                val_set = config["val"]
                # ★★★ 训练集需要展开 ★★★
                train_set_expanded = train_set * num_repeat
                all_folds.append((train_set_expanded, val_set))
        return all_folds
    else:
        raise ValueError(f"tag must be 'intra' or 'cross', but got: {tag}")


def get_dataset_info(dataset_name):
    """
    根据数据集名称返回该数据集的采样率 fs 和视频长度 video_len(单位为秒)

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
        "DLCN": {"fs": 30, "video_len": 60},
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
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels, filenames  = data
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
def evaluate(model, data_loader, device, foldidx, args, fs, plot_path):
    model.eval()

    hr_pred_all = []  # 累计预测mae
    hr_label_all = []
    snr_all = []  # 累计预测snr
    r_all = []

    data_loader = tqdm(data_loader, file=sys.stdout)
    val_len = args.val_len * fs
    for step, data in enumerate(data_loader):
        images, labels, filenames = data  # images:[B,C,T,H,W], T:[B,T]
        num_repeat = np.floor(images.shape[2] / val_len).astype(int)  # 计算 num_batches

        pred_all = []
        label_all = []

        for i in range(num_repeat):

            img_window = images[:, :, i * val_len:(i + 1) * val_len, :, :]
            label_window = labels[:, i * val_len:(i + 1) * val_len]

            pred = model(img_window.to(device))

            pred_np = pred.detach().cpu().numpy()
            label_np = label_window.detach().cpu().numpy()

            # if args.plot in ["wave", "both"]:
            #     fig_name = f'wave_{args.model_name}_{args.train_dataset}_scene{args.scene[0]}_{args.val_dataset}_scene{args.scene[1]}_fold{foldidx}_seed{args.seed}_aug{args.aug}_{step}_{i}'
            #     fig_path = os.path.join(plot_path, fig_name)
            #     plot_wave_psd(pred_np, label_np, fs, fig_path)

            pred_all.append(pred_np)
            label_all.append(label_np)

            hr_pred, hr_label, snr, r = calculate_metric_per_video(pred_np, label_np, fs,
                                                                   hr_method=args.hr_method)
            # print("hr_pred:{}, hr_label:{}".format(hr_pred, hr_label))
            hr_pred_all.append(hr_pred)
            hr_label_all.append(hr_label)
            snr_all.append(snr)
            r_all.append(r)

        pred_all = np.array(pred_all).flatten()
        label_all = np.array(label_all).flatten()

        signal_save_path = args.signal_path + f"/{args.model_name}/{args.train_dataset}_scene{args.scene[0]}_{args.val_dataset}_scene{args.scene[1]}_seed{args.seed}_aug{args.aug}"
        os.makedirs(signal_save_path, exist_ok=True)
        np.savez(
            os.path.join(signal_save_path, f'signal_{filenames[0]}.npz'),
            pred=pred_all,
            label=label_all
        )   


        data_loader.desc = 'val on {}'.format(step)

    np.savez(
        os.path.join(signal_save_path, f'hr_fold{foldidx}.npz'),
        hr_pred=hr_pred_all,
        hr_label=hr_label_all
    )  
    # if args.plot in ["blandaltman", "both"]:
    #     fig_name = f'blandaltman_{args.model_name}_{args.train_dataset}_scene{args.scene[0]}_{args.val_dataset}_scene{args.scene[1]}_fold{foldidx}__seed{args.seed}_aug{args.aug}_{val_len}'
    #     fig_path = os.path.join(plot_path, fig_name)
    #     plot_blandaltman(hr_pred_all, hr_label_all, fig_path)

    metrics_dict = calculate_metrics(hr_pred_all, hr_label_all, snr_all, r_all)

    return metrics_dict

# TODO 完成可视化结果图设计
def visualize_results(signal_dir, plot_path, args, max_plot_num=10, plot_length=None):
    """
    读取所有fold的预测心率和真实心率数据,并生成相应的可视化图
    :param fold_dir: 存储所有fold结果的文件夹路径
    :param plot_path: 可视化结果存储路径
    :param plot_length: wave可视化的长度
    :param max_plot_num: 可视化wave的个数
    """
    fs, video_len = get_dataset_info(args.val_dataset)
    val_len = fs * args.val_len
    plot_length = val_len

    # 如果选择Bland-Altman图，则调用plot_blandaltman函数
    if args.plot in "blandaltman, all":
        hr_pred_all = []
        hr_label_all = []
        
        # 遍历所有fold的结果文件
        for foldidx in range(5):  # 假设5折交叉验证
            npz_file = os.path.join(signal_dir, f'hr_fold{foldidx}.npz')  # 请确保路径和文件名的正确性
            if os.path.exists(npz_file):
                data = np.load(npz_file)
                hr_pred = data['hr_pred']
                hr_label = data['hr_label']

                # 将预测值和真实值加入到列表中
                hr_pred_all.extend(hr_pred.flatten())  # 展平并追加
                hr_label_all.extend(hr_label.flatten())  # 展平并追加
        fig_name = f'blandaltman_{args.model_name}_{args.train_dataset}_scene{args.scene[0]}_{args.val_dataset}_scene{args.scene[1]}_seed{args.seed}_aug{args.aug}_vallen{val_len}'
        plot_blandaltman(hr_pred_all, hr_label_all, fig_path=os.path.join(plot_path, fig_name))
        print("Bland-Altman plot generated for all folds!")
        
    # TODO 完善保存的文件名
    if args.plot in "wave, all":
        file_list = sorted([
            f for f in os.listdir(signal_dir)
            if f.startswith("signal") and f.endswith(".npz")
        ])

        for filename in file_list[:max_plot_num]:
            filepath = os.path.join(signal_dir, filename)
            data = np.load(filepath)
            pred = data['pred']
            label = data['label']

            # 对 pred 插值，长度对齐
            if len(pred) != len(label):
                x_pred = np.linspace(0, 1, len(pred))
                x_label = np.linspace(0, 1, len(label))
                pred = np.interp(x_label, x_pred, pred)

            # 截断到指定长度
            if plot_length is not None:
                pred = pred[:plot_length]
                label = label[:plot_length]

            sample_name = os.path.splitext(filename)[0]
            sample_id = sample_name.split('_')[1]  # 得到 '1'
            save_name = f'wave_{args.model_name}_{args.train_dataset}_scene{args.scene[0]}_{args.val_dataset}_scene{args.scene[1]}_seed{args.seed}_aug{args.aug}_vallen{val_len}_{sample_id}'

            fig_path = os.path.join(plot_path, save_name)
            plot_wave_psd(pred, label, fps=fs, fig_path=fig_path)

        print(f"Wave plot generated for {min(max_plot_num, len(file_list))} samples.")


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
            "value": np.mean(value_list),
            "std": np.std(value_list, ddof=1),         # 5个fold的 value 的 std
        }

    print("5-fold cross validation summary:")
    for metric, vals in final_metrics.items():
        print(f"{metric}: {vals['value']:.3f}, +/- {vals['std']:.3f}")

    return final_metrics


if __name__ == '__main__':
    flods = prepare_split_data('DLCN',  seed=42, scene='Raw', tag='intra')


