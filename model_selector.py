import sys

def select_model(model_name:str='PhysNet' ,len:int=160):

    if model_name == 'rpnet':
        from models.RPNet import RPNet
        model = RPNet()
    elif model_name == 'PhysNet':
        from models.PhysNet import PhysNet_padding_Encoder_Decoder_MAX
        model = PhysNet_padding_Encoder_Decoder_MAX(len)
    elif model_name == 'RhythmFormer':
        from models.RhythmFormer import RhythmFormer
        model = RhythmFormer()
    elif model_name == 'EfficientPhys':
        from models.EfficientPhys import EfficientPhys
        model = EfficientPhys()
    elif model_name == 'TSCAN':
        from models.TSCAN import TSCAN
        model = TSCAN()
    elif model_name == 'DeepPhys':
        from models.DeepPhys import DeepPhys
        model = DeepPhys()
    elif model_name == 'PhysFormer':
        from models.PhysFormer import ViT_ST_ST_Compact3_TDC_gra_sharp
        model = ViT_ST_ST_Compact3_TDC_gra_sharp(image_size=(len, 128, 128), patches=(4, 4, 4), dim=96, ff_dim=144,
                                           num_heads=4, num_layers=12, dropout_rate=0.1, theta=0.7)
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
