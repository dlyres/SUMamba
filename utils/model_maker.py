from model.plfa_model import make_model as PLFA
from model.MS1D_CNN import make_model as MS1D_CNN
from model.CCNN import make_model as CCNN
from model.UnetMamba import make_model as UnetMamba
from model.SSVEPFormer import make_model as SSVEPFormer


def make_model(args):
    model = None
    if args.model_name == 'PLFA':
        model = PLFA(args)
    if args.model_name == 'MS1D_CNN':
        model = MS1D_CNN(args)
    if args.model_name == 'CCNN':
        model = CCNN(args)
    if args.model_name == 'UnetMamba':
        model = UnetMamba(args)
    if args.model_name == 'SSVEPFormer':
        model = SSVEPFormer(args)


    return model