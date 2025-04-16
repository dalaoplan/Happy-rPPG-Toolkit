import sys

def select_model(model_name:str='PhysNet' ,len:int=160):

    if model_name == 'RhythmFormer':
        from models.RhythmFormer import RhythmFormer
        model = RhythmFormer()
    elif model_name == 'PhysNetback1':
        from models.PhyNetback1 import PhysNetback1
        model = PhysNetback1()

    else:
        print('Could not find model specified.')
        sys.exit(-1)

    return model


def select_loss(loss_name:str='ftLoss', fs:int=30):
    if loss_name == 'RhythmFormer':
        from Loss.LossComputer import RhythmFormer_Loss
        Loss = RhythmFormer_Loss()
    if loss_name == 'ftLoss':
        from Loss.ftLoss import MyLoss
        Loss = MyLoss(fs)
    if loss_name == 'NegPearson':
        from Loss.neg_pearson import NegPearsonLoss
        Loss = NegPearsonLoss()
    else:
        print('Could not find loss specified.')
        sys.exit(-1)

    return Loss
