import os


if __name__ == '__main__':

    scen = {'R', 'E', 'FIFP', 'VIFP', 'FIVP', 'VIVP'}  # ,
    model_name = {'GREEN', 'LGI', 'PBV'}      #, 'ICA', 'LGI', 'CHROME', 'GREEN', 'PBV''ICA', 'CHROME',
    test_dataset = 'DLCN'

    for model in model_name:
        for s in scen:
            # 训练
            command = f"python traditionalmethods\Predict.py  --test-dataset {test_dataset} --scen {s} --method-name {model}"

            # 测试
            # command = f"python test.py --train-dataset {train} --test-dataset {test} --model-name {model_name}"

            os.system(command)
            print(f"{model} test on {test_dataset} with {s} scen done!!!")
