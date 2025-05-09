import yaml
import os
import tempfile


def load_and_modify_config(yaml_path, changes: dict):
    """读取并修改 YAML 配置"""
    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # 修改参数
    for key, value in changes.items():
        cfg[key] = value

    return cfg


def run_train_with_config(cfg: dict):
    """将修改后的配置写入临时文件并调用训练脚本"""
    with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.yaml') as tmp:
        yaml.dump(cfg, tmp)
        tmp_path = tmp.name

    # 构造命令并调用
    command = f"python train.py --config {tmp_path}"
    os.system(command)


if __name__ == '__main__':
    config_path = 'config/intra.yaml'
    model_name = 'PhysNet'
    train_dataset = 'UBFCrPPG'
    val_dataset = 'DLCN'
    # 多个训练任务的设置
    train_scenes = ['Raw'] # ['rest', 'exercise', 'Raw']
    val_scenes = ['Raw']
    # TODO: 这里可以对 traindataset 和 val_dataset 进行循环
    for train_scene in train_scenes:
        for val_scene in val_scenes:
            # 修改参数
            changes = {
                'scene': [train_scene, val_scene],
                'model_name': model_name,
                'train_dataset': train_dataset,
                'val_dataset': val_dataset
            }
                # print(f"\n🌟 当前训练场景: {scene}")
            cfg = load_and_modify_config(config_path, changes)
            run_train_with_config(cfg)
            print(f"✅ {model_name}在{train_dataset}的{train_scene}场景中训练， 在{val_dataset}的{val_scene}场景中验证完成")
            print("=" * 50)

        print("🎉 所有训练任务已完成！")
