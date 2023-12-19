from networks.unet import UNet
# from networks.scribbleVC import scribbleVC

def net_factory(net_type="unet", in_chns=1, class_num=3, img_size=384, feature_zoom_size=48):
    if net_type == "unet":
        net = UNet(in_chns=in_chns, class_num=class_num).cuda()
    else:
        net = None
    return net
