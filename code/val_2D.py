import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import os
from torch.cuda.amp import autocast


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        recall = metric.binary.recall(pred, gt)
        precision = metric.binary.precision(pred, gt)
        jc = metric.binary.jc(pred, gt)
        specificity = metric.binary.specificity(pred, gt)
        ravd = metric.binary.ravd(pred, gt)
        return dice, hd95, recall, precision, jc, specificity, ravd
    else:
        return 0, 0, 0, 0, 0, 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256], amp=False):
    image, label = image.squeeze(1).squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(1).squeeze(0).cpu().detach().numpy()
    # ###odoc val
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        
        slice = image
            # x, y = slice.shape[0], slice.shape[1]
            # slice = zoom(
            #     slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
                0).float().cuda()
        net.eval()
        with torch.no_grad():
            with autocast(enabled=amp):
                out = torch.argmax(torch.softmax(
                    net(input)[0], dim=1), dim=1).squeeze(1).squeeze(0)
                out = out.cpu().detach().numpy()
                    # pred = zoom(
                    #     out, (x / patch_size[0], y / patch_size[1]), order=0)
                prediction = out
    # ###faz val
    elif len(image.shape) == 2:
        prediction = np.zeros_like(label)
        
        slice = image
            # x, y = slice.shape[0], slice.shape[1]
            # slice = zoom(
            #     slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            with autocast(enabled=amp):
                out = torch.argmax(torch.softmax(
                    net(input)[0], dim=1), dim=1).squeeze(1).squeeze(0)
                out = out.cpu().detach().numpy()
                    # pred = zoom(
                    #     out, (x / patch_size[0], y / patch_size[1]), order=0)
                prediction = out
    metric_list = []
    for i in range(1, classes):
        if i==1:
            metric_list.append(calculate_metric_percase(
                prediction == 1, label == 1))
        else:
            metric_list.append(calculate_metric_percase(
                prediction >= 1, label >= 1))
    return metric_list
def test_single_volume_ddp(accelerator,image, label, net, classes, patch_size=[256, 256], amp=False):
    image, label = image.squeeze(1).squeeze(0).detach(
    ), label.detach()
    # ###odoc val
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        
        # slice = image
            # x, y = slice.shape[0], slice.shape[1]
            # slice = zoom(
            #     slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = image.unsqueeze(
                0).float()
        net = accelerator.unwrap_model(net)
        net.eval()
        with torch.no_grad():
            with autocast(enabled=amp):
                out = torch.argmax(torch.softmax(
                    net(input)[0], dim=1), dim=1)
                out = out.detach()
                prediction = out
    # ###faz val
    elif len(image.shape) == 2:
        prediction = np.zeros_like(label)
        
        # slice = image
            # x, y = slice.shape[0], slice.shape[1]
            # slice = zoom(
            #     slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = image.unsqueeze(
            0).unsqueeze(0).float()
        net.eval()
        with torch.no_grad():
            with autocast(enabled=amp):
                out = torch.argmax(torch.softmax(
                    net(input)[0], dim=1), dim=1)
                out = out.detach()
                    # pred = zoom(
                    #     out, (x / patch_size[0], y / patch_size[1]), order=0)
                prediction = out
    # prediction, label = prediction.to(accelerator.device), label.to(accelerator.device)
    # prediction, label = prediction.to_dense(), label.to_dense()
    # prediction, label = accelerator.gather_for_metrics((prediction.contiguous(),label))
    metric_list = []
    label = label.squeeze(1).squeeze(0).cpu().numpy()
    prediction = prediction.squeeze(1).squeeze(0).cpu().numpy()
    for i in range(1, classes):
        if i==1:
            metric_list.append(calculate_metric_percase(
                prediction == 1, label == 1))
        else:
            metric_list.append(calculate_metric_percase(
                prediction >= 1, label >= 1))
    return metric_list
def test_single_volume_modularity(image, label, net, classes, patch_size=[256, 256], amp=False):
    image, label = image.squeeze(1).squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(1).squeeze(0).cpu().detach().numpy()
    # ###odoc val
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        
        slice = image
            # x, y = slice.shape[0], slice.shape[1]
            # slice = zoom(
            #     slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
                0).float().cuda()
        net.eval()
        with torch.no_grad():
            with autocast(enabled=amp):
                out = torch.argmax(torch.softmax(
                    net(input)[0], dim=1), dim=1).squeeze(1).squeeze(0)
                out = out.cpu().detach().numpy()
                    # pred = zoom(
                    #     out, (x / patch_size[0], y / patch_size[1]), order=0)
                prediction = out
    # ###faz val
    elif len(image.shape) == 2:
        prediction = np.zeros_like(label)
        
        slice = image
            # x, y = slice.shape[0], slice.shape[1]
            # slice = zoom(
            #     slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            with autocast(enabled=amp):
                out = torch.argmax(torch.softmax(
                    net(input)[0], dim=1), dim=1).squeeze(1).squeeze(0)
                out = out.cpu().detach().numpy()
                    # pred = zoom(
                    #     out, (x / patch_size[0], y / patch_size[1]), order=0)
                prediction = out
    metric_list = []
    for i in range(1, classes):
        if i==1:
            metric_list.append(calculate_metric_percase(
                prediction == 1, label == 1))
        else:
            metric_list.append(calculate_metric_percase(
                prediction >= 1, label >= 1))
    return metric_list

def test_single_volume_interpret_noise(image, label, net, classes, patch_size=[256, 256], amp=False):
    image, label = image.squeeze(1).squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(1).squeeze(0).cpu().detach().numpy()
    # ###odoc val
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        
        slice = image
            # x, y = slice.shape[0], slice.shape[1]
            # slice = zoom(
            #     slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
                0).float().cuda()
        net.eval()
        with torch.no_grad():
            with autocast(enabled=amp):
                out = torch.argmax(torch.softmax(
                    net(input)[0], dim=1), dim=1).squeeze(1).squeeze(0)
                out = out.cpu().detach().numpy()
                    # pred = zoom(
                    #     out, (x / patch_size[0], y / patch_size[1]), order=0)
                prediction = out
    # ###faz val
    elif len(image.shape) == 2:
        prediction = np.zeros_like(label)
        
        slice = image
            # x, y = slice.shape[0], slice.shape[1]
            # slice = zoom(
            #     slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            with autocast(enabled=amp):
                out = torch.argmax(torch.softmax(
                    net(input)[0], dim=1), dim=1).squeeze(1).squeeze(0)
                out = out.cpu().detach().numpy()
                    # pred = zoom(
                    #     out, (x / patch_size[0], y / patch_size[1]), order=0)
                prediction = out
    pre_label_clean = (label == prediction)
    label_wrong = (prediction + 1) * (~pre_label_clean) - 1
    label_wrong[label_wrong < 0] = 0
    metric_list = []
    for i in range(1, classes):
        if i==1:
            metric_list.append(calculate_metric_percase(
                prediction == 1, label_wrong == 1))
        else:
            metric_list.append(calculate_metric_percase(
                prediction >= 1, label_wrong >= 1))
    return metric_list

def test_single_volume_interpret_el(image, noisy_label, gt, net, classes, patch_size=[256, 256], amp=False):
    image, noisy_label, gt = image.squeeze(1).squeeze(0).cpu().detach(
    ).numpy(), noisy_label.squeeze(1).squeeze(0).cpu().detach().numpy(), gt.squeeze(1).squeeze(0).cpu().detach().numpy()
    # ###odoc val
    if len(image.shape) == 3:
        prediction = np.zeros_like(noisy_label)
        
        slice = image
            # x, y = slice.shape[0], slice.shape[1]
            # slice = zoom(
            #     slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
                0).float().cuda()
        net.eval()
        with torch.no_grad():
            with autocast(enabled=amp):
                out = torch.argmax(torch.softmax(
                    net(input)[0], dim=1), dim=1).squeeze(1).squeeze(0)
                out = out.cpu().detach().numpy()
                    # pred = zoom(
                    #     out, (x / patch_size[0], y / patch_size[1]), order=0)
                prediction = out
    # ###faz val
    elif len(image.shape) == 2:
        prediction = np.zeros_like(noisy_label)
        
        slice = image
            # x, y = slice.shape[0], slice.shape[1]
            # slice = zoom(
            #     slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            with autocast(enabled=amp):
                out = torch.argmax(torch.softmax(
                    net(input)[0], dim=1), dim=1).squeeze(1).squeeze(0)
                out = out.cpu().detach().numpy()
                    # pred = zoom(
                    #     out, (x / patch_size[0], y / patch_size[1]), order=0)
                prediction = out

    noise_gt = (gt == noisy_label)
    gt_notin_noise = (gt + 1) * (~noise_gt) - 1
    gt_notin_noise[gt_notin_noise < 0] = 0
    noise_notin_gt = (noisy_label + 1) * (~noise_gt) - 1
    noise_notin_gt[noise_notin_gt < 0 ] = 0
    prediction = (prediction + 1) *(~noise_gt) - 1
    prediction[prediction < 0]=0 


    metric_list = []
    for i in range(1, classes):
        if i==1:
            metric_list.append(calculate_metric_percase(
                prediction == 1, gt_notin_noise == 1))
        else:
            metric_list.append(calculate_metric_percase(
                prediction >= 1, gt_notin_noise >= 1))
    return metric_list


def test_single_volume_interpret_memorization(image, noisy_label, gt, net, classes, patch_size=[256, 256], amp=False):
    image, noisy_label, gt = image.squeeze(1).squeeze(0).cpu().detach(
    ).numpy(), noisy_label.squeeze(1).squeeze(0).cpu().detach().numpy(), gt.squeeze(1).squeeze(0).cpu().detach().numpy()
    # ###odoc val
    if len(image.shape) == 3:
        prediction = np.zeros_like(noisy_label)
        
        slice = image
            # x, y = slice.shape[0], slice.shape[1]
            # slice = zoom(
            #     slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
                0).float().cuda()
        net.eval()
        with torch.no_grad():
            with autocast(enabled=amp):
                out = torch.argmax(torch.softmax(
                    net(input)[0], dim=1), dim=1).squeeze(1).squeeze(0)
                out = out.cpu().detach().numpy()
                    # pred = zoom(
                    #     out, (x / patch_size[0], y / patch_size[1]), order=0)
                prediction = out
    # ###faz val
    elif len(image.shape) == 2:
        prediction = np.zeros_like(noisy_label)
        
        slice = image
            # x, y = slice.shape[0], slice.shape[1]
            # slice = zoom(
            #     slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            with autocast(enabled=amp):
                out = torch.argmax(torch.softmax(
                    net(input)[0], dim=1), dim=1).squeeze(1).squeeze(0)
                out = out.cpu().detach().numpy()
                    # pred = zoom(
                    #     out, (x / patch_size[0], y / patch_size[1]), order=0)
                prediction = out

    noise_gt = (gt == noisy_label)
    gt_notin_noise = (gt + 1) * (~noise_gt) - 1
    gt_notin_noise[gt_notin_noise < 0] = 0
    noise_notin_gt = (noisy_label + 1) * (~noise_gt) - 1
    noise_notin_gt[noise_notin_gt < 0 ] = 0
    prediction = (prediction + 1) *(~noise_gt) - 1
    prediction[prediction < 0]=0 


    metric_list = []
    for i in range(1, classes):
        if i==1:
            metric_list.append(calculate_metric_percase(
                prediction == 1, noise_notin_gt == 1))
        else:
            metric_list.append(calculate_metric_percase(
                prediction >= 1, noise_notin_gt >= 1))
    return metric_list



def test_single_volume_ds(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            slice = zoom(
                slice, (patch_size[0] / x, patch_size[1] / y), order=0)
            input = torch.from_numpy(slice).unsqueeze(
                0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                output_main, _, _, _ = net(input)
                out = torch.argmax(torch.softmax(
                    output_main, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                pred = zoom(
                    out, (x / patch_size[0], y / patch_size[1]), order=0)
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            output_main, _, _, _ = net(input)
            out = torch.argmax(torch.softmax(
                output_main, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list


def test_single_volume_cct(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(1).squeeze(0).cpu().detach().numpy(), label.squeeze(1).squeeze(0).cpu().detach().numpy()
    # print("image,shape=",image.shape)
    # print("label.shape=",label.shape)
    # print(image.shape)
    # print(label.shape)
    # image=image.transpose(0,3,1,2)
    # label=label.transpose(0,3,1,2)
    # label=label[:,0,:,:]
    # if len(image.shape) == 4:
    #     prediction = np.zeros_like(label)
    #     for ind in range(image.shape[0]):
    #         slice = image[ind,:, :, :]
    #         # x, y = slice.shape[2], slice.shape[3]
    #         # slice = zoom(
    #         #     slice, 1, order=0)
    #         input = torch.from_numpy(slice).unsqueeze(
    #             0).float().cuda()
    #         # input = torch.from_numpy(slice).float().cuda() 
    #         net.eval()
    #         with torch.no_grad():
    #             output_main = net(input)[0]
    #             # print(output_main.shape)
    #             out = torch.argmax(torch.softmax(
    #                 output_main, dim=1), dim=1).squeeze(0)
    #             # print(out.shape)
    #             prediction = out.cpu().detach().numpy()
    #             # pred = zoom(
    #             #     out, (x / patch_size[0], y / patch_size[1]), order=0)
    # else:
    #     # input = torch.from_numpy(image).unsqueeze(
    #         # 0).unsqueeze(0).float().cuda()

    #     input = torch.from_numpy(image).float().cuda()    
    #     net.eval()
    #     with torch.no_grad():
    #         # output_main, _, _, _ = net(input)
            
    #         output_main, o2 = net(input)
    #         print("output_main=",output_main.shape)
    #         print("o2=",o2.shape)

    #         out = torch.argmax(torch.softmax(
    #             output_main, dim=1), dim=1).squeeze(0)
    #         prediction = out.cpu().detach().numpy()

    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        
        slice = image
            # x, y = slice.shape[0], slice.shape[1]
            # slice = zoom(
            #     slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
                0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(
                net(input)[0], dim=1), dim=1).squeeze(1).squeeze(0)
            out = out.cpu().detach().numpy()
                # pred = zoom(
                #     out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction = out
    # ###faz val
    elif len(image.shape) == 2:
        prediction = np.zeros_like(label)
        
        slice = image
            # x, y = slice.shape[0], slice.shape[1]
            # slice = zoom(
            #     slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(
                net(input)[0], dim=1), dim=1).squeeze(1).squeeze(0)
            out = out.cpu().detach().numpy()
                # pred = zoom(
                #     out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction = out
    metric_list = []
    for i in range(1, classes):
        if i==1:
            metric_list.append(calculate_metric_percase(
                prediction == 1, label == 1))
        else:
            metric_list.append(calculate_metric_percase(
                prediction >= 1, label >= 1))
    return metric_list

def test_single_volume_tel(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(1).squeeze(0).cpu().detach().numpy(), label.squeeze(1).squeeze(0).cpu().detach().numpy()
    # print("image,shape=",image.shape)
    # print("label.shape=",label.shape)
    # print(image.shape)
    # print(label.shape)
    # image=image.transpose(0,3,1,2)
    # label=label.transpose(0,3,1,2)
    # label=label[:,0,:,:]
    # if len(image.shape) == 4:
    #     prediction = np.zeros_like(label)
    #     for ind in range(image.shape[0]):
    #         slice = image[ind,:, :, :]
    #         # x, y = slice.shape[2], slice.shape[3]
    #         # slice = zoom(
    #         #     slice, 1, order=0)
    #         input = torch.from_numpy(slice).unsqueeze(
    #             0).float().cuda()
    #         # input = torch.from_numpy(slice).float().cuda() 
    #         net.eval()
    #         with torch.no_grad():
    #             output_main = net(input)[0]
    #             # print(output_main.shape)
    #             out = torch.argmax(torch.softmax(
    #                 output_main, dim=1), dim=1).squeeze(0)
    #             # print(out.shape)
    #             prediction = out.cpu().detach().numpy()
    #             # pred = zoom(
    #             #     out, (x / patch_size[0], y / patch_size[1]), order=0)
    # else:
    #     # input = torch.from_numpy(image).unsqueeze(
    #         # 0).unsqueeze(0).float().cuda()

    #     input = torch.from_numpy(image).float().cuda()    
    #     net.eval()
    #     with torch.no_grad():
    #         # output_main, _, _, _ = net(input)
            
    #         output_main, o2 = net(input)
    #         print("output_main=",output_main.shape)
    #         print("o2=",o2.shape)

    #         out = torch.argmax(torch.softmax(
    #             output_main, dim=1), dim=1).squeeze(0)
    #         prediction = out.cpu().detach().numpy()

    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        
        slice = image
            # x, y = slice.shape[0], slice.shape[1]
            # slice = zoom(
            #     slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
                0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(
                net(input)[0], dim=1), dim=1).squeeze(1).squeeze(0)
            out = out.cpu().detach().numpy()
                # pred = zoom(
                #     out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction = out
    # ###faz val
    elif len(image.shape) == 2:
        prediction = np.zeros_like(label)
        
        slice = image
            # x, y = slice.shape[0], slice.shape[1]
            # slice = zoom(
            #     slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(
                net(input)[0], dim=1), dim=1).squeeze(1).squeeze(0)
            out = out.cpu().detach().numpy()
                # pred = zoom(
                #     out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction = out
    metric_list = []
    for i in range(1, classes):
        if i==1:
            metric_list.append(calculate_metric_percase(
                prediction == 1, label == 1))
        else:
            metric_list.append(calculate_metric_percase(
                prediction >= 1, label >= 1))
    return metric_list
        

            
    # metric_list = []
    # # print(prediction.shape)
    # # print(label.shape)
    # for i in range(1, classes):
    #     if i==1:
    #         metric_list.append(calculate_metric_percase(
    #             prediction == 1, label == 1))
    #     else:
    #         metric_list.append(calculate_metric_percase(
    #             prediction >= 1, label >= 1))
    # return metric_list
