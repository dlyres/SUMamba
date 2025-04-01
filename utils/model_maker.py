from model.plfa_model import make_model as PLFA
from model.VGT import make_model as VGT
from model.VIT import make_model as VIT
from model.SSGFormer import make_model as SSGFormer
from model.MS1D_CNN import make_model as MS1D_CNN
from model.CCNN import make_model as CCNN
from model.VIM import make_model as VIM
from model.SEMamba import make_model as SEMamba
from model.SSGFormerSEMamba import make_model as SSGFormerSEMamba
from model.SSGFormerxlstm import make_model as SSGFormerxlstm
from model.SEMambaSA import make_model as SEMambaSA
from model.MambaSE import make_model as MambaSE
from model.Unet import make_model as Unet
from model.UnetMamba import make_model as UnetMamba
from model.SpaMambaxlstm import make_model as SpaMambaxlstm
from model.SSVEPFormer import make_model as SSVEPFormer
from model.DCNN import make_model as DCNN
from model.Unet_GRU import make_model as UnetGRU
from model.Unet_LSTM import make_model as UnetLSTM


def make_model(args):
    model = None
    if args.model_name == 'PLFA':
        model = PLFA(args)
    if args.model_name == 'VGT':
        model = VGT(args)
    if args.model_name == 'SSGFormer':
        model = SSGFormer(args)
    if args.model_name == 'MS1D_CNN':
        model = MS1D_CNN(args)
    if args.model_name == 'CCNN':
        model = CCNN(args)
    if args.model_name == 'VIT':
        model = VIT(args)
    if args.model_name == 'VIM':
        model = VIM(args)
    if args.model_name == 'SEMamba':
        model = SEMamba(args)
    if args.model_name == 'SEMambaSA':
        model = SEMambaSA(args)
    if args.model_name == 'SSGFormerSEMamba':
        model = SSGFormerSEMamba(args)
    if args.model_name == 'SSGFormerxlstm':
        model = SSGFormerxlstm(args)
    if args.model_name == 'MambaSE':
        model = MambaSE(args)
    if args.model_name == 'Unet':
        model = Unet(args)
    if args.model_name == 'SpaMambaxlstm':
        model = SpaMambaxlstm(args)
    if args.model_name == 'UnetMamba':
        model = UnetMamba(args)
    if args.model_name == 'SSVEPFormer':
        model = SSVEPFormer(args)
    if args.model_name == '3DCNN':
        model = DCNN(args)
    if args.model_name == 'Unet_GRU':
        model = UnetGRU(args)
    if args.model_name == 'Unet_LSTM':
        model = UnetLSTM(args)

    return model