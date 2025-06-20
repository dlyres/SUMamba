import argparse
import os
import time
import numpy as np
import torch
import drawing.show_acc as sc
import drawing.show_auc as auc
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler
from utils.model_maker import *
from utils.train_views import *
from utils.calculate_itr import calculate
from dataloader.Dataloader import SSVEPDataset, make_dataloader
from sklearn.model_selection import KFold
from torch.utils.data import Subset


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # create dataset and dataloader
    dataset = SSVEPDataset(args.dataset_name, cross_validation=args.cross_validation)

    data_size = len(dataset)
    print(f"total dataset size:{data_size}")

    # check available GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"training GPU: {gpu_name}")
    else:
        print("no available GPU, using CPU")

    model = make_model(args)
    kfold = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
    acc_list = []
    auc_list = []
    itr_list = []

    # k-fold validation
    for fold, (train_indices, val_indices) in enumerate(kfold.split(range(data_size))):
        num_fold = (fold + 1) / args.k_folds
        print(f"Fold {num_fold}")

        train_subset = torch.utils.data.Subset(dataset, train_indices)
        val_subset = torch.utils.data.Subset(dataset, val_indices)
        print(f'train set size:{len(train_subset)}')
        print(f'validation set size:{len(val_subset)}')

        train_dataloader = make_dataloader(train_subset, batch_size=args.batch_size, shuffle=True)
        val_dataloader = make_dataloader(val_subset, batch_size=args.batch_size, shuffle=True)

        scheduler = None

        if args.optim == 'SGD':
            optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5E-5)
        else:
            optim = torch.optim.Adam(model.parameters(), lr=args.lr)

        if args.isReduceLR:
            scheduler = lr_scheduler.ReduceLROnPlateau(optim,
                                                       mode=args.mode,
                                                       patience=args.patience,
                                                       factor=args.factor,
                                                       verbose=True)
        writer = SummaryWriter(f"tensorboard/{args.dataset_name}/cross_validation/{args.model_name}/{num_fold}")

        train_step = 0
        test_step = 0
        final_class_acc = np.zeros(args.num_classes)  # final acc of each class
        total_acc_list = []  # acc trend while training
        start_time = time.time()
        model_auc = 0

        # early stopping mechanism
        last_lr = 0
        temp_lr = 0
        lr_count = 0

        
        for epoch in range(args.epochs):
            total_train_loss = train_one_epoch(model=model,
                                               optimizer=optim,
                                               data_loader=train_dataloader,
                                               device=device,
                                               epoch=epoch,
                                               in_c=args.in_c,
                                               dataset_name=args.dataset_name)
            total_test_loss, class_acc, total_acc, auc, itr_list_batch = evaluate(model=model,
                                                                                  data_loader=val_dataloader,
                                                                                  device=device,
                                                                                  epoch=epoch,
                                                                                  in_c=args.in_c,
                                                                                  dataset_name=args.dataset_name,
                                                                                  num_classes=args.num_classes)
            model_auc = auc
            total_acc_list.append(total_acc)

            if args.isReduceLR:
                last_lr = optim.param_groups[0]['lr']
                scheduler.step(total_test_loss)
                temp_lr = optim.param_groups[0]['lr']
                if last_lr != temp_lr:
                    lr_count = lr_count + 1

            loss_scalar = f"{args.dataset_name}: model={args.model_name}, " \
                          f"fold={num_fold}, in_c={args.in_c}, optim={args.optim}, lr={args.lr}, bs={args.batch_size}, " \
                          f"epoch={args.epochs}, isReduceLR={args.isReduceLR}"

            if args.model_name == 'CCNN' or args.model_name == 'plfa_model' or args.model_name == 'SSVEPFormer_model':
                loss_scalar = loss_scalar + f", patch_size={args.patch_size}, embed_dim={args.embed_dim}, " \
                                            f"depth={args.depth}, num_heads={args.num_heads}"

            writer.add_scalar(f"{loss_scalar}/train_loss", total_train_loss, train_step)
            writer.add_scalar(f"{loss_scalar}/test_loss", total_test_loss, test_step)
            train_step = train_step + 1
            test_step = test_step + 1
            final_class_acc = class_acc

            if(lr_count == 5):
                time_itr = sum(itr_list_batch) / len(val_subset) + 1.5
                itr = calculate(time_itr, total_acc, args.num_classes)
                itr_list.append(itr)
                break

        total_acc_list = np.round(np.array(total_acc_list), decimals=4)
        final_class_acc = np.round(final_class_acc, decimals=4)


        end_time = time.time() - start_time
        print("Time spent on {} model training：{:.2f} minutes".format(args.model_name, end_time / 60))

        save_dir = f'weights/cross_validation/{args.dataset_name}/{args.model_name}{num_fold}.pth'
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        torch.save(model.state_dict(), save_dir)

        writer.close()

        if args.accuracy and args.num_models == 1:
            sc.show(total_acc_list, final_class_acc, args, num_fold=num_fold)

        acc_list.append(total_acc_list[-1])
        auc_list.append(model_auc)
        model = make_model(args)

    itr_list = np.round(np.array(itr_list), decimals=2)

    avg_acc = sum(acc_list) / args.k_folds
    avg_auc = sum(auc_list) / args.k_folds
    avg_itr = sum(itr_list) / args.k_folds
    print(f'acc of each fold: {acc_list}, avg acc: {avg_acc:.4f}')
    print(f'auc of each fold: {auc_list}, avg auc: {avg_auc:.4f}')
    print(f'itr of each fold: {itr_list}, avg itr: {avg_itr:.2f}')

    if args.num_models != 1:
        return acc_list, avg_acc, auc_list, avg_auc
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # training parameters
    parser.add_argument('--dataset_name', type=str, default='BETA')
    parser.add_argument('--num_classes', type=int, default=40)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--optim', type=str, default='Adam')
    parser.add_argument('--k_folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cross_validation', type=bool, default=True)

    # learning rate decay parameters
    parser.add_argument('--isReduceLR', type=bool, default=True)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--factor', type=float, default=0.5)
    parser.add_argument('--mode', type=str, default='min')
    parser.add_argument('--verbose', type=bool, default=True)

    parser.add_argument('--model_name', type=str, default='UnetMamba')

    parser.add_argument('--num_models', type=int, default=1)

    # model parameters
    parser.add_argument('--in_c', type=int, default=2)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--in_c_spa', type=int, default=4)
    parser.add_argument('--depth_spa', type=int, default=6)
    parser.add_argument('--kernel_size_spa', type=tuple, default=(7, 7))

    parser.add_argument('--accuracy', type=bool, default=False)
    parser.add_argument('--auc', type=bool, default=False)

    args = parser.parse_args()

    if args.num_models == 1:
        main(args)
    else:
        model_list = ['CCNN', 'MS1D_CNN', 'plfa']
        model_acc_cross = []
        model_auc_cross = []
        for model_name in model_list:
            args.model_name = model_name
            acc_list, acc_avg, auc_list, auc_avg = main(args)
            acc_list.append(acc_avg)
            model_acc_cross.append(acc_list)
            auc_list.append(auc_avg)
            model_auc_cross.append(auc_list)
            print(f"{args.model_name} models train finished")
        if args.accuracy:
            sc.show_cross(args, model_acc_cross, model_list)
        if args.auc:
            auc.show(args, model_auc_cross, model_list)
