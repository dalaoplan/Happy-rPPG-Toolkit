import os
from utils import get_dataset_info


if __name__ == '__main__':

    scen = {'R', 'E', 'FIFP', 'VIFP', 'FIVP', 'VIVP'} # {'R', 'E', 'FIFP', 'VIFP', 'FIVP', 'VIVP'}
    model_name = {'ICA', 'CHROME', 'POS'}      #'ICA', 'LGI', 'CHROME', 'GREEN', 'PBV', 'POS'
    test_dataset = {'DLCN'}  #'DLCN', 'PURE', 'UBFCrPPG', 'COHFACE'

    for model in model_name:
        for test in test_dataset:
            fs, videolen = get_dataset_info(test)
            for s in scen:
                # 训练
                # command = f"python traditionalmethods\Predict.py  --test-dataset {test_dataset} --scen {s} --method-name {model}"

                # 测试
                command = f"python Predict.py --method-name {model} --test-dataset {test} --scen {s} --fps {fs}"

                os.system(command)
                print(f"{model} test on {test} with {s} scen done!!!")
