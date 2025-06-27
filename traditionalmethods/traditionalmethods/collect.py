import os
import json
import pandas as pd


def collect_results_from_json(json_dir):
    """
    从指定文件夹中读取所有 .json 文件，并解析其中的实验结果，
    返回一个 pandas.DataFrame，包含模型名称、训练集、测试集，以及各指标（MAE, RMSE, MAPE, Pearson, SNR, MACC）。
    """
    rows = []
    # 遍历文件夹
    for file_name in os.listdir(json_dir):
        if file_name.endswith(".json"):
            json_path = os.path.join(json_dir, file_name)
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 解析 JSON 中的关键信息
            model_name = data.get("method_name", "UnknownModel")
            test_dataset = data.get("test_dataset", "UnknownTest")
            Scen = data.get("scene", "UnknownTest")
            metrics = data.get("metrics", {})

            # 获取各指标的 value 和 std
            mae_val = metrics.get("MAE", {}).get("value", None)
            mae_std = metrics.get("MAE", {}).get("std", None)

            rmse_val = metrics.get("RMSE", {}).get("value", None)
            rmse_std = metrics.get("RMSE", {}).get("std", None)

            mape_val = metrics.get("MAPE", {}).get("value", None)
            mape_std = metrics.get("MAPE", {}).get("std", None)

            pearson_val = metrics.get("Pearson", {}).get("value", None)
            pearson_std = metrics.get("Pearson", {}).get("std", None)

            snr_val = metrics.get("SNR", {}).get("value", None)
            snr_std = metrics.get("SNR", {}).get("std", None)

            macc_val = metrics.get("MACC", {}).get("value", None)
            macc_std = metrics.get("MACC", {}).get("std", None)

            # 拼接成 “值±标准差” 的字符串（如果没有 std，则只显示值）
            def val_std_str(val, std):
                if val is None:
                    return ""
                if std is None:
                    return f"{val:.4f}"
                return f"{val:.4f}±{std:.4f}"

            row = {
                "Model(Scen)/Test": f"{model_name}({Scen})/{test_dataset}",
                "MAE": val_std_str(mae_val, mae_std),
                "RMSE": val_std_str(rmse_val, rmse_std),
                "MAPE": val_std_str(mape_val, mape_std),
                "Pearson": val_std_str(pearson_val, pearson_std),
                "SNR": val_std_str(snr_val, snr_std),
                "MACC": val_std_str(macc_val, macc_std),
            }
            rows.append(row)

    # 将所有结果转换为 DataFrame
    df = pd.DataFrame(rows, columns=[
        "Model(Scen)/Test", "MAE", "RMSE", "MAPE", "Pearson", "SNR", "MACC"
    ])
    return df


if __name__ == "__main__":
    # 假设所有 .json 文件都存放在 'weight' 文件夹中
    json_dir = r"D:\LZP\Happy-rPPG-Toolkit\traditionalmethods\result\save"
    df_results = collect_results_from_json(json_dir)

    # 打印 DataFrame
    print("实验结果表格：")
    print(df_results.to_string(index=False))

    # 如果想以 Markdown 格式打印，可以使用：
    # print(df_results.to_markdown(index=False))

    # 如果想保存为 CSV：
    df_results.to_csv("all_experiment_results.csv", index=False, encoding="utf-8-sig")
