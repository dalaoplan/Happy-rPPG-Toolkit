import os
import sys
import argparse
import torch
import json
import numpy as np
from tqdm import tqdm
from post_process import calculate_metric_per_video, calculate_metrics
from visual import plot_wave_psd, plot_blandaltman
from MyDataset import MyDataset
from utils import read_split_data
from CHROME import CHROME_DEHAAN
from GREEN import GREEN
from ICA_POH import ICA_POH
from LGI import LGI
from PBV import PBV
from POS_WANG import POS_WANG


def test(args):

    print(f'Method {args.method_name} test on {args.test_dataset} in scene {args.scene}')
    plot_path = f"result/Plots"
    if args.plot:
        if os.path.exists(plot_path) is False:
            os.makedirs(plot_path)


    test_len = args.test_len * args.fps
    test_path = read_split_data(args.test_dataset, args.scene)

    dataset = MyDataset(data_list=test_path,
                            T=test_len,
                            transform=None,
                             method='test')


    test_loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=args.nw,
                                             collate_fn=dataset.collate_fn)



    hr_pred_all = []  # 累计预测mae
    hr_label_all= []
    snr_all = []  # 累计预测snr
    r_all = []

    data_loader = tqdm(test_loader, file=sys.stdout)


    for step, data in enumerate(data_loader):
        images, labels = data  # images:[B,C,T,H,W], T:[B,T]
        images = images.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
        labels = labels.squeeze(0).cpu().numpy()
        
        num_repeat = np.floor(images.shape[0] / test_len).astype(int)  # 计算 num_batches

        # pred_all = []
        # label_all = []

        for i in range(num_repeat):

            img_window = images[i*test_len:(i+1)*test_len]
            label = labels[i*test_len:(i+1)*test_len]
            
            if args.method_name == "POS":
                pred = POS_WANG(img_window, args.fps)
            elif args.method_name == "CHROME":
                pred = CHROME_DEHAAN(img_window, args.fps)
            elif args.method_name == "ICA":
                pred = ICA_POH(img_window, args.fps)
            elif args.method_name == "GREEN":
                pred = GREEN(img_window)
            elif args.method_name == "LGI":
                pred = LGI(img_window)
            elif args.method_name == "PBV":
                pred = PBV(img_window)
            else:
                raise ValueError("unsupervised method name wrong!")
            
            if args.plot in ["wave", "both"]:
                fig_name = f'wave_{args.method_name}_{args.test_dataset}_{step + 1}_{i + 1}'
                fig_path = os.path.join(plot_path, fig_name)
                plot_wave_psd(pred, label, args.fps, fig_path)

            # pred_all.append(pred_np)
            # label_all.append(label_np)

            hr_pred, hr_label, snr, r = calculate_metric_per_video(pred, label, args.fps, hr_method=args.hr_method)
            # print("hr_pred:{}, hr_label:{}".format(hr_pred, hr_label))
            hr_pred_all.append(hr_pred)
            hr_label_all.append(hr_label)
            snr_all.append(snr)
            r_all.append(r)


        # pred_all = np.array(pred_all)
        # label_all = np.array(label_all)

        data_loader.desc = 'Test on {}'.format(step)

    if args.plot in ["blandaltman", "both"]:
        fig_name = f'blandaltman_{args.method_name}_{args.scene}_{args.test_dataset}_{test_len}'
        fig_path = os.path.join(plot_path, fig_name)

        plot_blandaltman(hr_pred_all, hr_label_all, fig_path)

    metrics_dict = calculate_metrics(hr_pred_all, hr_label_all, snr_all, r_all)


    # 准备保存的测试信息
    test_results = {
        "method_name": args.method_name,
        "scene": args.scene,
        "test_dataset": args.test_dataset,
        "test_len": test_len,
        "visualization_path": plot_path,  # 你可以指定一个路径
        "test_num_samples": len(test_path),
        "metrics": metrics_dict  # 假设 calculate_metrics 返回一个字典
}


    # 确保保存目录存在
    save_dir = "result/save"
    os.makedirs(save_dir, exist_ok=True)

    # 生成保存文件的路径
    save_path = os.path.join(save_dir, f"{args.method_name}_{args.scene}_{args.test_dataset}_{test_len}.json")

    # 保存到 JSON 文件
    with open(save_path, "w") as f:
        json.dump(test_results, f, indent=4)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-dataset', type=str, default='UBFCPhys', help='test dataset name')
    parser.add_argument('--scene', type=str, default='Raw', help='test scene, R: Relax, E: Exercise, FIFP, VIFP, FIVP, VIVP')
    parser.add_argument('--test-len', type=int, default=10, help='test length, 10 second')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--nw', type=int, default=0, help='num_workers')
    parser.add_argument('--hr_method', type=str, default='FFT', help='=calculate hr use FFT or Peak')
    parser.add_argument('--plot', type=str, default='blandaltman', help='wave: only plot wave fig., blandaltman: only plot blandaltman fig., both: plot two fig.')
    parser.add_argument('--method-name', default='CHROME', help='create model name')


    opt = parser.parse_args()

    test(opt)



