import argparse
import logging
import os
import random
import shutil
import sys
import time
import math
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from dataloaders import utils
from dataloaders.dataset_procns import BaseDataSets, RandomGenerator
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from val_2D import test_single_volume, test_single_volume_ds,calculate_metric_percase
import matplotlib.pyplot as plt

# ###ProCNS lib
from utils.iou_computation import update_metric_stat, compute_iou, compute_dice, iter_iou_stat, get_mask, iter_fraction_pixelwise
# ###
from scipy.interpolate import splrep, splev
# ###ProCNS Function
from scipy.optimize import curve_fit
from scipy.interpolate import splrep, splev
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
# ###
# ### Noisy Learning
from utils.seg_loss.ce import GeneralizedCELoss
import pandas as pd
from utils.PRSALoss_PersamplePrototype_MultiLayers import AffinityLoss, AffinityLoss_PixelsAssignment






parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/FAZ_h5', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='odoc_bs_12/Unet_pCE', help='experiment_name')
parser.add_argument('--sup_type', type=str,
                    default='mask', help='supervision label type(scr ; label ; scr_n ; keypoint ; block)')
parser.add_argument('--model', type=str,
                    default='unet_head', help='model_name')
parser.add_argument('--num_classes', type=int,  default=2,
                    help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--in_chns', type=int, default=3,
                    help='image channel')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size',nargs='+', type=int, default=[384,384],
                    help='patch size of network input')
parser.add_argument('--gpus', type=int, default=0,
                    help='gpu index,must set CUDA_VISIBLE_DEVICES at terminal')
parser.add_argument('--img_class', type=str,
                    default='faz', help='the img class(odoc or faz)')
parser.add_argument('--seed', type=int,  default=2022, help='random seed')
parser.add_argument('--amp', type=bool,  default=0,
                        help='whether use amp training')
parser.add_argument('--period_iter', type=int, default=10)
parser.add_argument('--eval_interval', type=int, default=5, help='eval per iterations')
# parser.add_argument('--thr_iter', type=int, default=6000)
parser.add_argument('--thr_epoch', type=int, default=15)
parser.add_argument('--warm_up_epoch', type=int, default=5)
parser.add_argument('--thr_conf', type=float, default=0.8)
parser.add_argument('--thr_conf_correction', type=float, default=0.6)
parser.add_argument('--alpha', type=float, default=0.2)
parser.add_argument('--r_threshold', type=float, default=0.9)
parser.add_argument('--img_size', type=int, default=384)
# parser.add_argument('--pretrain_iter', type=int, default=100)
parser.add_argument('--epoch_update_interval', type=int, default=100, help='update dataset per epoch_nums')
args = parser.parse_args()



def png_save(x_data, y_data, epoch, save_path, class_idx, data_source='lwlr'):
    png_path = save_path + '/fit_png/'
    csv_path = save_path +'/metric_csv/'
    if not os.path.exists(png_path):
        os.makedirs(png_path)
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
    print("x_data.shape",x_data.shape)
    print("x_data.shape",y_data.shape)
    df_list = []
    x_data_list = x_data.tolist()
    y_data_list = y_data.tolist()
    df_list.append(x_data_list)
    df_list.append(y_data_list)
    df = pd.DataFrame(df_list)
    df.to_csv(csv_path+data_source+'_class{}_epoch{}.csv'.format(class_idx ,epoch), index=False)

    return

# ### gaussian k
def Gaussian_kernel_bandwidth(y_data):
    print("y_data.shape = ",y_data.shape)
    params = {'bandwidth': np.logspace(-0.3, 0, 50)}
    kde = KernelDensity(kernel='gaussian')
    grid = GridSearchCV(kde, params)
    grid.fit(y_data)
    print(grid.best_params_)
    return grid.best_params_


# ### LWLR
class LWLR:

    def fit(self, X, y):
        self.X = np.hstack((np.ones((X.shape[0], 1)), X.reshape(-1,1)))
        self.y = y.reshape(-1,1)
        self.k = Gaussian_kernel_bandwidth(self.y)['bandwidth']

    def predict(self, x):
        x = np.hstack((1, x))
        weights = np.eye(self.X.shape[0])
        for i in range(self.X.shape[0]):
            diff = x - self.X[i]
            weights[i][i] = np.exp(diff.dot(diff.T) / (-2.0 * self.k ** 2))
        xt = self.X.T
        beta = np.linalg.inv(xt.dot(weights).dot(self.X)).dot(xt).dot(weights).dot(self.y)
        return x.dot(beta)

###
# ####ProCNS When
def curve_func(x, a, b, c):
    return a * (1 - np.exp(-1 / c * x ** b))


def fit(func, x, y):
    popt, pcov = curve_fit(func, x, y, p0=(1, 1, 1), method='trf', sigma=np.geomspace(1, .1, len(y)),
                           absolute_sigma=True, bounds=([0, 0, 0], [1, 1, np.inf]))
    return tuple(popt)


def derivation(x, a, b, c):
    x = x + 1e-6  # numerical robustness
    return a * b * 1 / c * np.exp(-1 / c * x ** b) * (x ** (b - 1))


def label_update_epoch(ydata_fit, lwlr, class_idx, threshold=0.9, eval_interval=100, num_iter_per_epoch=10581 / 10, save_path='/png/'):
    xdata_fit = np.linspace(0, len(ydata_fit) * eval_interval, len(ydata_fit))
    ydata_fit[0] = 0
    epoch_num = len(ydata_fit) * eval_interval / num_iter_per_epoch
    lwlr.fit(xdata_fit, ydata_fit)
    y_data_lwlr_fit = np.zeros_like(xdata_fit)
    for i, x_i in enumerate(xdata_fit):
        y_data_lwlr_fit[i] = lwlr.predict(x_i)
    y_data_lwlr_fit[0] = 0
    print("y_data_lwlr_fit.shape = ",y_data_lwlr_fit.shape)
    png_save(xdata_fit, ydata_fit, epoch_num, save_path, class_idx,data_source='origin')
    png_save(xdata_fit, y_data_lwlr_fit, epoch_num, save_path, class_idx,data_source='lwlr')
    
    a, b, c = fit(curve_func, xdata_fit, y_data_lwlr_fit)
    epoch = np.arange(1, epoch_num)
    # y_hat = curve_func(epoch, a, b, c)
    # png_save(epoch.tolist(), y_hat.tolist(), epoch_num, save_path, class_idx,data_source='sigmodalfit')
    relative_change = abs(abs(derivation(epoch, a, b, c)) - abs(derivation(1, a, b, c))) / abs(derivation(1, a, b, c))
    relative_change[relative_change > 1] = 0
    update_epoch = np.sum(relative_change <= threshold) + 1
    return update_epoch  # , a, b, c


def if_update(iou_value, current_epoch, lwlr, class_idx, threshold=0.90, batch_per_epoch=100, eval_interval=5, save_path='/png/'):
    update_epoch = label_update_epoch(iou_value, lwlr, class_idx, threshold=threshold, eval_interval=eval_interval ,num_iter_per_epoch=batch_per_epoch, save_path = save_path)
    return current_epoch >= update_epoch  # , update_epoch

# ###ANPM masking the noise regions
# pseudo_label:seg; prediction_argmax; prediction_max_prob:
def update_allclass(idx, sample, IoU_npl_indx, prediction_argmax, prediction_max_prob, mask_threshold, IoU_npl_constraint, class_constraint=True, update_or_mask='update', update_all_bg_img=False):
    seg_label = sample['mask']
    seg_label = np.expand_dims(seg_label, axis=0)


    seg_argmax = prediction_argmax  # num_class,h,w
    seg_prediction_max_prob = prediction_max_prob  # 1,h,w
    b, h, w = seg_label.shape  # 1,h,w
    if 0 in np.unique(seg_label) and len(np.unique(seg_label))==1:
        return 
    # if seg label does not belong to the set of class that needs to be updated (exclude the background class), return
    if set(np.unique(seg_label)).isdisjoint(set(IoU_npl_indx[1:])):
        if update_all_bg_img and not (set(np.unique(seg_label))-set(np.array([0,255]))):
            pass
        else:
            return 

    # if the class_constraint==True and seg label has foreground class
    # we prevent using predicted class that is not in the pseudo label to correct the label
    if class_constraint == True and (set(np.unique(seg_label[0])) - set(np.array([0, 255]))):
        for i_batch in range(b):
            seg_label = torch.from_numpy(seg_label).long()
            seg_argmax = torch.from_numpy(seg_argmax).long()
            seg_prediction_max_prob = torch.from_numpy(prediction_max_prob)
            unique_class = torch.unique(seg_label[i_batch])
            print("unique_class=", unique_class)
            indx = torch.zeros((h, w), dtype=torch.long)
            for element in unique_class:
                indx = indx | (seg_argmax[i_batch] == element)
            seg_argmax[i_batch][(indx == 0)] = 255

    seg_mask_255 = (seg_argmax == 255)

    # seg_change_indx means which pixels need to be updated,
    # find index where prediction is different from label,
    # and  it is not a ignored index and confidence is larger than threshold
    seg_change_indx = (seg_label != seg_argmax) & (~seg_mask_255) & (
            seg_prediction_max_prob > mask_threshold)

    # when set to "both", only when predicted class and pseudo label both existed in the set, the label would be corrected
    # this is a conservative way, during our whole experiments, IoU_npl_constraint is always set to be "single",
    # this is retained here in case user may find in useful for their dataset
    if IoU_npl_constraint == 'both':
        class_indx_seg_argmax = torch.zeros((b, h, w), dtype=torch.bool)
        class_indx_seg_label = torch.zeros((b, h, w), dtype=torch.bool)

        for element in IoU_npl_indx:
            class_indx_seg_argmax = class_indx_seg_argmax | (seg_argmax == element)
            class_indx_seg_label = class_indx_seg_label | (seg_label == element)
        seg_change_indx = seg_change_indx & class_indx_seg_label & class_indx_seg_argmax

    #  when set to "single", when predicted class existed in the set, the label would be corrected, no need to consider pseudo label
    # e.g. when person belongs to the set, motor pixels in the pseudo label can be updated to person even if motor is not in set
    elif IoU_npl_constraint == 'single':
        class_indx_seg_argmax = torch.zeros((b, h, w), dtype=torch.bool)

        for element in IoU_npl_indx:
            class_indx_seg_argmax = class_indx_seg_argmax | (seg_argmax == element)
        seg_change_indx = seg_change_indx & class_indx_seg_argmax

    # if the foreground class portion is too small, do not update
    seg_label_clone = seg_label.clone()
    seg_label_clone[seg_change_indx] = seg_argmax[seg_change_indx]
    if torch.sum(seg_label_clone!=0) < 0.5 * torch.sum(seg_label!=0) and torch.sum(seg_label_clone==0)/(b*h*w)>0.95:
        return

    # update or mask 255
    if update_or_mask == 'update':
        seg_label[seg_change_indx] = seg_argmax[seg_change_indx]  # update all class of the pseudo label
    else:
        # mask the pseudo label for 255 without computing the loss
        seg_label[seg_change_indx] = (args.num_classes*torch.ones((b, h, w), dtype=torch.long))[
            seg_change_indx]  # the updated pseudo label

    return seg_label.cpu().numpy()
# ####



def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    
    model = net_factory(net_type=args.model, in_chns=args.in_chns, class_num=num_classes)
    db_train = BaseDataSets(base_dir=args.root_path, split="train", transform=transforms.Compose([
        RandomGenerator(args.patch_size,img_class=args.img_class)
    ]), sup_type=args.sup_type, img_class=args.img_class)
    db_val = BaseDataSets(base_dir=args.root_path,
                           split="val", img_class=args.img_class)   
    lwlr = LWLR() 
  
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=0)

    model.train()
    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)

    ce_loss = CrossEntropyLoss(ignore_index=num_classes)
    
    agenergy_loss = AffinityLoss()
    agenergy_loss_pixassignment = AffinityLoss_PixelsAssignment()
    param = {
        'loss_softmax':True,
        'loss_gce_q':0.5,
    }
    gce_loss = GeneralizedCELoss(param)
    dice_loss = losses.pDLoss(num_classes, ignore_index=args.num_classes)
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))
    
# ProCNS Previous_updated_class_list
    Updated_class_list = []

    IoU_npl_dict = {}
    for i in range(num_classes):
        IoU_npl_dict[i] = []
    IoU_npl_indx = [0]


    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    performance = 0.0
    loss_agenergy_kernels_desc = [{"weight": 1, "xy": 6, "rgb": 0.1}]
    loss_agenergy_radius = 5
    batch_per_epoch = len(trainloader)
    iterator = tqdm(range(max_epoch), ncols=70)
    thr_iter = args.thr_epoch * batch_per_epoch
    # update_status_once = False
    for epoch_num in iterator:
        TP_npl = [0] * num_classes
        P_npl = [0] * num_classes
        T_npl = [0] * num_classes
        for i_batch, sampled_batch in enumerate(trainloader):
            if args.img_class == 'faz':
                volume_batch, label_batch, weight_batch, correction = sampled_batch['image'].unsqueeze(1), sampled_batch['sup_label'], sampled_batch['weight'], sampled_batch['mask']
                volume_batch, label_batch, weight_batch, correction = volume_batch.cuda(), label_batch.cuda(), weight_batch.cuda(), correction.cuda()
            elif args.img_class == 'odoc' or args.img_class == 'polyp':
                volume_batch, label_batch, weight_batch, correction = sampled_batch['image'], sampled_batch['sup_label'], sampled_batch['weight'], sampled_batch['mask']
                volume_batch, label_batch , weight_batch, correction = volume_batch.cuda(), label_batch.cuda(), weight_batch.cuda(), correction.cuda()
    
            out = model(volume_batch)
            if args.model == 'fcnet':
                high_feats, outputs = out
            elif args.model in ['deeplabv3plus', 'treefcn']:
                outputs, _, high_feats = out
            elif args.model == 'unet_head':
                outputs, feature, de1, de2, de3, de4, high_feats1, high_feats2 = out
            elif args.model == 'unet_cg':
                outputs, feature, de1, de2, de3, de4, concept = out
            elif args.model == 'unet_cg_de4':
                outputs, feature, de1, de2, de3, bilinear_de4, concept = out
            else:
                outputs, feature, de1, de2, de3, de4 = out
            loss_ce = ce_loss(outputs, label_batch[:].long())
            outputs_soft = torch.softmax(outputs, dim=1)
            if iter_num < thr_iter:
                unlabeled_RoIs_sup = (sampled_batch['sup_label'] == args.num_classes).unsqueeze(1).repeat(1, args.num_classes, 1, 1)
                unlabeled_RoIs_sup = unlabeled_RoIs_sup.cuda()
                # print("unlabeled_RoIs.unique", unlabeled_RoIs.unique())
                if args.img_class == 'faz':
                    three_channel = volume_batch.repeat(1, 3, 1, 1)
                elif args.img_class == 'odoc' or args.img_class == 'polyp':
                    three_channel = volume_batch
                three_channel = three_channel.cuda()
                out_agenergy = agenergy_loss(
                    feature[-2].detach(),
                    de3.detach(),
                    label_batch,
                    outputs,
                    loss_agenergy_kernels_desc,
                    loss_agenergy_radius,
                    three_channel,
                    args.img_size,
                    args.img_size,
                    unlabeled_RoIs_sup,
                    args.num_classes
                )
                out_agenergy_loss = out_agenergy["loss"]
                loss = loss_ce + 0.1 * out_agenergy_loss
            else:

                noisy_label = correction[:].long()

                unlabeled_RoIs = (sampled_batch['mask'] == args.num_classes).unsqueeze(1).repeat(1, args.num_classes, 1, 1)
                unlabeled_RoIs = unlabeled_RoIs.cuda()
                unlabeled_RoIs_sup = (sampled_batch['sup_label'] == args.num_classes).unsqueeze(1).repeat(1, args.num_classes, 1, 1)
                unlabeled_RoIs_sup = unlabeled_RoIs_sup.cuda()
                

                if args.img_class == 'faz':
                    three_channel = volume_batch.repeat(1, 3, 1, 1)
                elif args.img_class == 'odoc' or args.img_class == 'polyp':
                    three_channel = volume_batch
                three_channel = three_channel.cuda()
                
                out_agenergy = agenergy_loss_pixassignment(
                    feature[-2].detach(),
                    de3.detach(),
                    noisy_label,
                    outputs,
                    loss_agenergy_kernels_desc,
                    loss_agenergy_radius,
                    three_channel,
                    args.img_size,
                    args.img_size,
                    unlabeled_RoIs,
                    unlabeled_RoIs_sup,
                    args.num_classes
                )
                out_agenergy_loss = out_agenergy["loss"]
                bounder_diceloss = out_agenergy["bounder_loss"]
                outputs_rd = out_agenergy["prediction_redefined"]
                outputs_rd_soft = torch.softmax(
                        outputs_rd, dim=1)


                heated_map = out_agenergy["heated_map"]


                loss_pse_sup = ce_loss(outputs, noisy_label)
                loss = loss_ce + 0.1 * out_agenergy_loss + 0.01 * bounder_diceloss + 0.5 * loss_pse_sup
# ##


            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - (iter_num - thr_iter) / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            iter_num = iter_num + 1
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            # writer.add_scalar('info/loss_tree', out_tree_loss, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            
            if iter_num < thr_iter:

                logging.info(
                    'iteration %d : loss : %f, loss_ce: %f, out_agenergy_loss: %f' %
                    (iter_num, loss.item(), loss_ce.item(), out_agenergy_loss.item()))

            
            if iter_num > thr_iter:

                logging.info(
                    'iteration %d : loss : %f, loss_ce: %f, bounder_dice_loss: %f, out_agenergy_loss: %f' %
                    (iter_num, loss.item(), loss_ce.item(), bounder_diceloss.item(), out_agenergy_loss.item()))
            writer.add_scalar('info/lr', lr_, iter_num)
            if iter_num % 5 == 0:
                image = volume_batch[0, :, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                outputs = torch.argmax(torch.softmax(
                            outputs, dim=1), dim=1, keepdim=True)
                outputs = outputs[0, ...] * 50
                if iter_num > thr_iter:
                    crt = correction[0, ...].unsqueeze(0)*50

                labs = label_batch[0, ...].unsqueeze(0) * 50
                if args.img_class == 'odoc' or args.img_class == 'polyp':
                    outputs, labs = outputs.repeat(3, 1, 1), labs.repeat(3, 1, 1)
                    if iter_num > thr_iter:
                        crt = crt.repeat(3, 1, 1)

                
                if iter_num > thr_iter:
                    writer.add_scalar('info/loss_agenergy', out_agenergy_loss, iter_num)
                    # writer.add_scalar('info/loss_psesup_dice', loss_pse_sup, iter_num)
                    writer.add_image('train/Image', image, iter_num)
                    writer.add_image('train/Supervised_Label', labs, iter_num,dataformats='CHW')
                    writer.add_image('train/Correction_Label', crt, iter_num, dataformats='CHW')
                    writer.add_image('train/Prediction',
                                outputs[0,...] * 50, iter_num,dataformats='HW')
                    writer.add_image('train/HeatedMap', heated_map[0,...] * 50, iter_num,dataformats='CHW')
            if iter_num > 0 and iter_num % 5 == 0:
                model.eval()
                metric_list = 0.0
                loss_val = 0.0
    
                for i_batch, sampled_batch in enumerate(valloader):
                    if args.img_class == 'faz':
                        volume_batch, label_batch = sampled_batch['image'].unsqueeze(1), sampled_batch['label']
                        volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
                    elif args.img_class == 'odoc' or args.img_class == 'polyp':
                        volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                        volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
                    # with torch.no_grad():
                    #     outputs_val = model(volume_batch)[0]
                    #     outputs_soft_val = torch.softmax(outputs_val, dim=1)
                    #     loss_ce_val = ce_loss(outputs_val, label_batch[:].long())
                    #     loss = 0.5 * (loss_ce_val + dice_loss(outputs_soft_val,
                    #         label_batch.unsqueeze(1)))
                    #     loss_val+=loss
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
                    metric_list = metric_list+np.array(metric_i)
                    
                # loss_val=loss_val/len(db_val)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)
                    writer.add_scalar('info/val_{}_recall'.format(class_i+1),
                                      metric_list[class_i, 2], iter_num)
                    writer.add_scalar('info/val_{}_precision'.format(class_i+1),
                                      metric_list[class_i, 3], iter_num)
                    writer.add_scalar('info/val_{}_jc'.format(class_i+1),
                                      metric_list[class_i, 4], iter_num)
                    writer.add_scalar('info/val_{}_specificity'.format(class_i+1),
                                      metric_list[class_i, 5], iter_num)
                    writer.add_scalar('info/val_{}_ravd'.format(class_i+1),
                                      metric_list[class_i, 6], iter_num)
                    # writer.add_scalar('info/total_loss_val', loss_val, iter_num)
    

                performance = np.mean(metric_list, axis=0)[0]
                mean_hd95 = np.mean(metric_list, axis=0)[1]
                mean_recall = np.mean(metric_list, axis=0)[2]
                mean_precision = np.mean(metric_list, axis=0)[3]
                mean_jc = np.mean(metric_list, axis=0)[4]
                mean_specificity = np.mean(metric_list, axis=0)[5]
                mean_ravd = np.mean(metric_list, axis=0)[6]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)
                writer.add_scalar('info/val_mean_recall', mean_recall, iter_num)
                writer.add_scalar('info/val_mean_precision', mean_precision, iter_num)
                writer.add_scalar('info/val_mean_jc', mean_jc, iter_num)
                writer.add_scalar('info/val_mean_specificity', mean_specificity, iter_num)
                writer.add_scalar('info/val_mean_ravd', mean_ravd, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                logging.info(
                    'iteration %d : mean_dice : %f mean_hd95 : %f : mean_recall : %f mean_precision : %f : mean_jc : %f mean_specificity : %f : mean_ravd : %f : total_loss : %f ' % (iter_num, performance, mean_hd95, mean_recall, mean_precision, mean_jc, mean_specificity, mean_ravd,loss_val))
                model.train()

            if iter_num % 100 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
            if iter_num % batch_per_epoch == 0 and iter_num < thr_iter:
                model.eval()
                logging.info("update weight start")

                ds = trainloader.dataset
                if not os.path.exists(os.path.join(snapshot_path, 'ensemble', str(iter_num))):
                    os.makedirs(os.path.join(snapshot_path,
                                'ensemble', str(iter_num)))
                # for idx, images in tqdm(ds.images.items(), total=len(ds)):
                for idx, images in ds.images.items():
                    img = images['image']
                    # img = zoom(
                    #     img, (256 / img.shape[0], 256 / img.shape[1]), order=0)
                    if args.img_class == 'faz':
                        img = torch.from_numpy(img).unsqueeze(
                            0).unsqueeze(0).cuda()
                    if args.img_class == 'odoc' or args.img_class == 'polyp':
                        img = torch.from_numpy(img).unsqueeze(
                            0).cuda()
                    with torch.no_grad():
                        pred = torch.softmax(model(img)[0], dim=1)
                    pred = pred.squeeze(0).cpu().numpy()
                    # pred = zoom(
                    #     pred, (1, images['image'].shape[0] / 256, images['image'].shape[1] / 256), order=0)
                    pred = torch.from_numpy(pred)
                    weight = torch.from_numpy(images['weight'])
                    if args.img_class == 'faz' or args.img_class == 'polyp':
                        x0, x1 = pred[0], pred[1]
                        weight[0, ...] = args.alpha * x0 + \
                            (1 - args.alpha) * weight[0, ...]
                        weight[1, ...]= args.alpha * x1 + \
                            (1 - args.alpha) * weight[1, ...]
                    elif args.img_class == 'odoc':
                        x0, x1, x2 = pred[0], pred[1], pred[2]
                        weight[0, ...] = args.alpha * x0 + \
                            (1 - args.alpha) * weight[0, ...]
                        weight[1, ...]= args.alpha * x1 + \
                            (1 - args.alpha) * weight[1, ...]
                        weight[2, ...] = args.alpha * x2 + \
                            (1 - args.alpha) * weight[2, ...]
                    trainloader.dataset.images[idx]['weight'] = weight.numpy()
                    trainloader.dataset.images[idx]['mask'] = np.argmax(weight.numpy(),axis=0,keepdims=False) 

                model.train()
                
            if iter_num > thr_iter and iter_num % args.eval_interval == 0:
                model.eval()
                pred_np = torch.argmax(outputs_soft, dim=1, keepdim=False).detach().cpu().numpy()
                label_np_updated = noisy_label.detach().cpu().numpy()
                TP_npl, P_npl, T_npl = update_metric_stat(pred_np, label_np_updated, TP_npl,P_npl, T_npl, num_classes=num_classes)
                IoU_npl = compute_dice(TP_npl, P_npl, T_npl, num_classes = num_classes)
                for i in range(num_classes):
                    IoU_npl_dict[i].append(IoU_npl[i])
                TP_npl = [0] * num_classes
                P_npl = [0] * num_classes
                T_npl = [0] * num_classes
                model.train()
            if iter_num >= max_iterations:
                break
        # epoch
        # noisy pseudo label fit
        if iter_num > thr_iter and (epoch_num - args.thr_epoch) > args.warm_up_epoch:
            model.eval()
            for class_idx in range(1, num_classes):
                # current code only support update each class once, if updated, it won't be updated again
                if not class_idx in Updated_class_list:
                    current_epoch = epoch_num - args.thr_epoch
                    update_sign = if_update(np.array(IoU_npl_dict[class_idx]), current_epoch, lwlr, class_idx, threshold=args.r_threshold, batch_per_epoch=batch_per_epoch, eval_interval= args.eval_interval, save_path = snapshot_path)
                    if update_sign:
                        IoU_npl_indx.append(class_idx)
                        Updated_class_list.append(class_idx)
            if len(IoU_npl_indx)>1:
                logging.info("-----updated class list-----")
                logging.info(Updated_class_list)
                logging.info("----------------------------")
            # if args.img_class == 'odoc':
            #     for indx, class_name in enumerate(
            #                 ['background', 'oc','od']):
            #         writer.add_scalar({'npl_' + class_name: IoU_npl[indx]}, iter_num)
            # if args.img_class == 'faz' or args.img_class == 'polyp' :
            #     for indx, class_name in enumerate(
            #                 ['background', 'foreground']):
            #         writer.add_scalar({'npl_' + class_name: IoU_npl[indx]}, step=epoch_num)
            if epoch_num % args.epoch_update_interval == 0 and len(IoU_npl_indx) > 1 and (epoch_num - args.thr_epoch) / args.epoch_update_interval >= 1:
                # update the segmentation label
                ds = trainloader.dataset
                metric_list_pseudo = 0.0
                logging.info("correct label start")
                for idx, sample in ds.images.items():
                    metric_list_pseudo_i = []
                    img = sample['image']
                    # img = zoom(
                    #     img, (256 / img.shape[0], 256 / img.shape[1]), order=0)
                    if args.img_class == 'faz':
                        img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).cuda()
                    if args.img_class == 'odoc' or args.img_class == 'polyp':
                        img = torch.from_numpy(img).unsqueeze(0).cuda()
                    with torch.no_grad():
                        pred = torch.softmax(model(img)[0], dim=1)
                        pred_argmax = torch.argmax(pred, dim=1, keepdim=False)
                        pred_max = torch.max(pred, dim=1, keepdim=False)[0]
                    prediction_max_prob = pred_max.cpu().numpy()
                    prediction_argmax = pred_argmax.cpu().numpy()
                    # print("prediction_max_prob.shape = ", prediction_max_prob.shape)
                    # print("prediction_argmax.shape = ", prediction_argmax.shape)
                    label_correction = update_allclass(idx, sample, IoU_npl_indx,  prediction_argmax, prediction_max_prob, args.thr_conf_correction, 'single', class_constraint=True, update_or_mask='no_update', update_all_bg_img=True)
                    # print("label_correction.shape = ", label_correction.shape)
                    # assert 255 not in np.unique(label_correction.cpu().numpy())
                    
                    if type(label_correction) != type(None):
                        trainloader.dataset.images[idx]['mask'] = label_correction.squeeze(0)
                    else:   
                        trainloader.dataset.images[idx]['mask'] = np.argmax(sample['weight'],axis=0,keepdims=False)
                    pseudo = trainloader.dataset.images[idx]['mask']
                    label = trainloader.dataset.images[idx]['gt']
                    for i in range(1, args.num_classes):
                        if i==1:
                            metric_list_pseudo_i.append(calculate_metric_percase(
                                pseudo == 1, label == 1))
                        else:
                            metric_list_pseudo_i.append(calculate_metric_percase(
                                pseudo >= 1, label >= 1))
                    metric_list_pseudo = metric_list_pseudo+np.array(metric_list_pseudo_i)
                metric_list_pseudo = metric_list_pseudo / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/pseudo_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/pseudo_{}_recall'.format(class_i+1),
                                      metric_list[class_i, 2], iter_num)
                    writer.add_scalar('info/pseudo_{}_precision'.format(class_i+1),
                                      metric_list[class_i, 3], iter_num)
                    # writer.add_scalar('info/total_loss_pseudo', loss_pseudo, iter_num)
    

                performance_pseudo = np.mean(metric_list, axis=0)[0]
                mean_recall_pseudo = np.mean(metric_list, axis=0)[2]
                mean_precision_pseudo = np.mean(metric_list, axis=0)[3]
                writer.add_scalar('info/pseudo_mean_dice', performance_pseudo, iter_num)
                writer.add_scalar('info/pseudo_mean_recall', mean_recall_pseudo, iter_num)
                writer.add_scalar('info/pseudo_mean_precision', mean_precision_pseudo, iter_num)
                    
            model.train()

                
            

            # the classes that need to be updated in the current epoch
            

        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)
    snapshot_path = "../model/{}/{}".format(
        args.exp, args.sup_type)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
