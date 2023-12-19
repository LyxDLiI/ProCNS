import argparse
from email.mime import image
import os
import re
import shutil
from tkinter import image_types
import pandas as pd
import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch, cv2
from medpy import metric
import random
# from scipy.ndimage import zoom
# from scipy.ndimage.interpolation import zoom
# from dataloaders.dataset import BaseDataSets, BaseDataSets_octa, RandomGenerator, RandomGeneratorv2, BaseDataSets_octasyn, RandomGeneratorv3, BaseDataSets_octasynv2, RandomGeneratorv4, train_aug, val_aug, BaseDataSets_octawithback, BaseDataSets_cornwithback,BaseDataSets_drivewithback,BaseDataSets_chasesyn, BaseDataSets_chasewithbackori, BaseDataSets_chasewithback
from tqdm import tqdm
from skimage.measure import label

# from networks.efficientunet import UNet
from networks.net_factory import net_factory

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ODOC_h5', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='odoc_bs_12/pCE_Unet', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=3,
                    help='output channel of network')
parser.add_argument('--in_chns', type=int, default=3,
                    help='image channel')
parser.add_argument('--sup_type', type=str, default="scribble",
                    help='label')
parser.add_argument('--img_class', type=str,
                    default='faz', help='the img class(odoc, faz or polyp)')

def get_client_ids(image_class,base_dir):
    
    if image_class == "faz":
        faz_test_set = 'FAZ/test/'+pd.Series(os.listdir(base_dir+"/FAZ/test"))
        faz_training_set = 'FAZ/train/'+pd.Series(os.listdir(base_dir+"/FAZ/train"))
        faz_test_set = faz_test_set.tolist()
        faz_training_set = faz_training_set.tolist()
        return [faz_training_set, faz_test_set]
    elif image_class == "odoc":
        odoc_test_set = 'ODOC/test/'+pd.Series(os.listdir(base_dir+"/ODOC/test"))
        odoc_training_set = 'ODOC/train/'+pd.Series(os.listdir(base_dir+"/ODOC/train"))
        odoc_test_set = odoc_test_set.tolist()
        odoc_training_set = odoc_training_set.tolist()
        return [odoc_training_set, odoc_test_set]
    elif image_class == "polyp":
        polyp_test_set = 'Polyp/test/'+pd.Series(os.listdir(base_dir+"/Polyp/test"))
        polyp_training_set = 'Polyp/train/'+pd.Series(os.listdir(base_dir+"/Polyp/train"))
        polyp_test_set = polyp_test_set.tolist()
        polyp_training_set = polyp_training_set.tolist()
        return [polyp_training_set, polyp_test_set]
       

    else:
        return "ERROR KEY"
def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        jaccard = metric.binary.jc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        assd = metric.binary.assd(pred, gt)
        se = metric.binary.sensitivity(pred, gt)
        sp = metric.binary.specificity(pred, gt)
        recall = metric.binary.recall(pred, gt)
        precision = metric.binary.precision(pred, gt)
        return dice, jaccard, hd95, assd, se, sp, recall, precision
    else:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

def test_single_image(case, net, test_save_path, FLAGS):

    h5f = h5py.File(FLAGS.root_path +
                            "/{}".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['mask'][:]
    prediction = np.zeros_like(label)

    if FLAGS.img_class == 'odoc':
        prediction = np.zeros_like(label)
        
        slice = image
        input = torch.from_numpy(slice).unsqueeze(
                0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(
                net(input)[0], dim=1), dim=1).squeeze(1).squeeze(0)
            out = out.cpu().detach().numpy()
            prediction = out
            item = case.split("/")[-1].split(".")[0]
            cv2.imwrite(test_save_path+'/pre/' + item + "_pred.png", prediction * 127.)
            cv2.imwrite(test_save_path + '/pre/' + item + "_gt.png", label * 127.)
    elif FLAGS.img_class == 'polyp':
        prediction = np.zeros_like(label)
        
        slice = image
        input = torch.from_numpy(slice).unsqueeze(
                0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(
                net(input)[0], dim=1), dim=1).squeeze(1).squeeze(0)
            out = out.cpu().detach().numpy()

            prediction = out

            item = case.split("/")[-1].split(".")[0]
            cv2.imwrite(test_save_path+'/pre/' + item + "_pred.png", prediction * 255.)
            cv2.imwrite(test_save_path + '/pre/' + item + "_gt.png", label * 255.)
    # ###faz val
    elif FLAGS.img_class == 'faz':
        prediction = np.zeros_like(label)
        
        slice = image
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(
                net(input)[0], dim=1), dim=1).squeeze(1).squeeze(0)
            out = out.cpu().detach().numpy()
            prediction = out
            item = case.split("/")[-1].split(".")[0]
            cv2.imwrite(test_save_path+'/pre/' + item + "_pred.png", prediction * 255.)
            cv2.imwrite(test_save_path + '/pre/' + item + "_gt.png", label * 255.)
    
    
    
    if FLAGS.img_class == 'faz' or FLAGS.img_class == 'polyp':
        metric = calculate_metric_percase(prediction == 1, label == 1)
        return metric
    if FLAGS.img_class == 'odoc':
        metric1 = calculate_metric_percase(prediction == 1, label == 1)
        metric2 = calculate_metric_percase(prediction >= 1, label >= 1)
        return metric1, metric2

def Inference(FLAGS):

    train_ids, test_ids = get_client_ids(FLAGS.img_class,FLAGS.root_path)
    image_list = []
    image_list = test_ids
    snapshot_path = "../model/{}{}/".format(
        FLAGS.exp,FLAGS.sup_type)

    test_save_path = "../model/{}test/{}/".format(
        FLAGS.exp,FLAGS.sup_type)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path+'/pre/')
    net = net_factory(net_type=FLAGS.model, in_chns=FLAGS.in_chns,
                      class_num=FLAGS.num_classes).cuda()

    save_mode_path = os.path.join(
        snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    net.load_state_dict(torch.load(save_mode_path),strict=False)
    print("init weight from {}".format(save_mode_path))
    net.eval()

    names = []
    dices = []
    jaccards = []
    HD95s = []
    ASSDs = []
    SEs = []
    SPs = []
    Recs = []
    Pres = []
    dices1 = []
    jaccards1 = []
    HD95s1 = []
    ASSDs1 = []
    SEs1 = []
    SPs1 = []
    Recs1 = []
    Pres1 = []
    dices2 = []
    jaccards2 = []
    HD95s2 = []
    ASSDs2 = []
    SEs2 = []
    SPs2 = []
    Recs2 = []
    Pres2 = []
    if FLAGS.img_class=='faz' or FLAGS.img_class=='polyp':
        for case in tqdm(image_list):
            print(case)
            

            metric = test_single_image(case, net, test_save_path, FLAGS)
            dice,jaccard,HD95,ASSD,SE,SP,Rec,Pre = metric[0],metric[1],metric[2],metric[3],metric[4],metric[5],metric[6],metric[7]
            
            names.append(str(case))
            dices.append(dice)
            jaccards.append(jaccard)
            HD95s.append(HD95)
            ASSDs.append(ASSD)
            SEs.append(SE)
            SPs.append(SP)
            Recs.append(Rec)
            Pres.append(Pre)
            
        import pandas as pd
        dataframe = pd.DataFrame({'name':names,'dice':dices,'jaccard':jaccards,'HD95':HD95s,'ASSD':ASSDs,'SE':SEs,'SP':SPs,'Rec':Recs,'Pre':Pres})
        dataframe.to_csv(test_save_path + "result.csv",index=False,sep=',')
        print('Counting CSV generated!')
        mean_std_resultframe = pd.DataFrame({'name':['mean','std'],'dice':[np.mean(dices),np.std(dices)],'jaccard':[np.mean(jaccards),np.std(jaccards)],'HD95':[np.mean(HD95s),np.std(HD95s)],'ASSD':[np.mean(ASSDs),np.std(ASSDs)],'SE':[np.mean(SEs),np.std(SEs)],'SP':[np.mean(SPs),np.std(SPs)],'Rec':[np.mean(Recs),np.std(Recs)],'Pre':[np.mean(Pres),np.std(Pres)]})
        mean_std_resultframe.to_csv(test_save_path + "mean_std_result.csv",index=False,sep=',')
        print('Mean and Std CSV generated!')
        avg_dice = np.mean(dices)
    if FLAGS.img_class=='odoc':
        for case in tqdm(image_list):
            print(case)
            

            metric1, metric2 = test_single_image(case, net, test_save_path,FLAGS)
            dice,jaccard,HD95,ASSD,SE,SP,Rec,Pre = metric1[0],metric1[1],metric1[2],metric1[3],metric1[4],metric1[5],metric1[6],metric1[7]
            names.append(str(case))
            dices1.append(dice)
            jaccards1.append(jaccard)
            HD95s1.append(HD95)
            ASSDs1.append(ASSD)
            SEs1.append(SE)
            SPs1.append(SP)
            Recs1.append(Rec)
            Pres1.append(Pre)
            dice,jaccard,HD95,ASSD,SE,SP,Rec,Pre = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            dice,jaccard,HD95,ASSD,SE,SP,Rec,Pre = metric2[0],metric2[1],metric2[2],metric2[3],metric2[4],metric2[5],metric2[6],metric2[7]
            dices2.append(dice)
            jaccards2.append(jaccard)
            HD95s2.append(HD95)
            ASSDs2.append(ASSD)
            SEs2.append(SE)
            SPs2.append(SP)
            Recs2.append(Rec)
            Pres2.append(Pre)
        import pandas as pd
        dataframe = pd.DataFrame({'name':names,'dice_cup':dices1,'jaccard_cup':jaccards1,'HD95_cup':HD95s1,'ASSD_cup':ASSDs1,'SE_cup':SEs1,'SP_cup':SPs1,'Rec_cup':Recs1,'Pre_cup':Pres1,'dice_disc':dices2,'jaccard_disc':jaccards2,'HD95_disc':HD95s2,'ASSD_disc':ASSDs2,'SE_disc':SEs2,'SP_disc':SPs2,'Rec_disc':Recs2,'Pre_disc':Pres2, 'Pre_disc':Pres2})
        dataframe.to_csv(test_save_path + "result.csv",index=False,sep=',')
        print('Counting CSV generated!')
        mean_std_resultframe = pd.DataFrame({'name':['mean','std'],'dice_cup':[np.mean(dices1),np.std(dices1)],'jaccard_cup':[np.mean(jaccards1),np.std(jaccards1)],'HD95_cup':[np.mean(HD95s1),np.std(HD95s1)],'ASSD_cup':[np.mean(ASSDs1),np.std(ASSDs1)],'SE_cup':[np.mean(SEs1),np.std(SEs1)],'SP_cup':[np.mean(SPs1),np.std(SPs1)],'Rec_cup':[np.mean(Recs1),np.std(Recs1)],'Pre_cup':[np.mean(Pres1),np.std(Pres1)],'dice_disc':[np.mean(dices2),np.std(dices2)],'jaccard_disc':[np.mean(jaccards2),np.std(jaccards2)],'HD95_disc':[np.mean(HD95s2),np.std(HD95s2)],'ASSD_disc':[np.mean(ASSDs2),np.std(ASSDs2)],'SE_disc':[np.mean(SEs2),np.std(SEs2)],'SP_disc':[np.mean(SPs2),np.std(SPs2)],'Rec_disc':[np.mean(Recs2),np.std(Recs2)],'Pre_disc':[np.mean(Pres2),np.std(Pres2)]})
        mean_std_resultframe.to_csv(test_save_path + "mean_std_result.csv",index=False,sep=',')
        print('Mean and Std CSV generated!')
        avg_dice = np.mean(dices1)
    return avg_dice
        


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    FLAGS = parser.parse_args()
    total = 0.0
    seed = 2022 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    random.seed(seed)  
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
        # for i in [5]:
    mean_dice = Inference(FLAGS)
    print("Test is Finished")

