import torch
import torch.nn.functional as F
from utils import losses, metrics, ramps
class AffinityLoss(torch.nn.Module):
    def forward(
            self, encoder_feature, decoder_feature, sup_label, y_hat, kernels_desc, kernels_radius, sample, height_input, width_input,Unlabeled_RoIs,num_classes
    ):
 
        # refined_prediction
        N, C, height_pred, width_pred = y_hat.shape
        device = y_hat.device
        _, _, height_encoder_feature, width_encoder_feature = encoder_feature.shape
        _, _, height_decoder_feature, width_decoder_feature = decoder_feature.shape
        # ###encoder prototype & feature map
        
        sup_label_encoder = F.interpolate(sup_label.unsqueeze(1).float(), size=(height_encoder_feature, width_encoder_feature), mode='nearest')
        sup_label_decoder = F.interpolate(sup_label.unsqueeze(1).float(), size=(height_decoder_feature, width_decoder_feature), mode='nearest')
        # print("sup_label_downsample.shape = ", sup_label_downsample.shape)
        batch_prototype = AffinityLoss._batch_prototype_generator(encoder_feature, sup_label_encoder, num_classes).repeat(1,N,1)#num,N,C1
        persample_prototype = AffinityLoss._persample_prototype_generator(decoder_feature, sup_label_decoder, num_classes)#num,N,C2
        prototype = torch.cat((batch_prototype, persample_prototype), dim=-1)
        # print("prototype.shape = ", prototype.shape)
        # prototype_l2 = F.normalize(batch_prototype, p=2, dim=2)#num_classes,N,C
        feature_encoder_upsample = F.interpolate(encoder_feature, size=(height_pred, width_pred), mode='bilinear', align_corners=False)#N,C,H,W
        feature_decoder_upsample = F.interpolate(decoder_feature, size=(height_pred, width_pred), mode='bilinear', align_corners=False)#N,C,H,W
        feature_upsample = torch.cat((feature_encoder_upsample, feature_decoder_upsample), dim=1)
        prototype_l2 = F.normalize(prototype, p=2, dim=2)
        affinity_map = AffinityLoss._affinity(feature_upsample, prototype_l2, num_classes)# N,num_class,H,W
        # ## redefined prediction(Affinity)
        y_redefined = affinity_map * y_hat
        y_hat_softmax=torch.softmax(y_redefined, dim=1)
        assert y_hat_softmax.dim() == 4, 'Prediction must be a NCHW batch'
        ###
        

        assert width_input % width_pred == 0 and height_input % height_pred == 0 and \
            width_input * height_pred == height_input * width_pred, \
            f'[{width_input}x{height_input}] !~= [{width_pred}x{height_pred}]'

        kernels = self._create_kernels(
            kernels_desc, kernels_radius, sample, N, height_pred, width_pred, device
        )
        # print(kernels.shape)
        # kernels = kernels * (Unlabeled_RoIs.unsqueeze(2).unsqueeze(2))

        # denom = N * height_pred * width_pred
        denom = Unlabeled_RoIs.sum()

        y_hat_unfolded = self._unfold(y_hat_softmax, kernels_radius)
        # kernels = kernels * (Unlabeled_RoIs.unsqueeze(2).unsqueeze(2))
        # print(kernels.shape)
        # print(y_hat_unfolded.shape)
        product_kernel_x_y_hat = (kernels * y_hat_unfolded) \
            .view(N, C, (kernels_radius * 2 + 1) ** 2, height_pred, width_pred) \
            .sum(dim=2, keepdim=False)


        # Using shortcut for Pott's class compatibility model
        loss = -(product_kernel_x_y_hat * y_hat_softmax * Unlabeled_RoIs).sum()
        # comment out to save computation, total loss may go below 0
        loss = kernels.sum() + loss
        loss = torch.clamp(loss, min=1e-5)


        out = {
            'loss': loss / denom, 'prediction_redefined':y_redefined, 'heated_map':product_kernel_x_y_hat
        }

        return out

    @staticmethod
    def _downsample(img, height_dst, width_dst):
        f_down = F.adaptive_avg_pool2d
        return f_down(img, (height_dst, width_dst))

    @staticmethod
    def _create_kernels(
            kernels_desc, kernels_radius, sample, N, height_pred, width_pred, device
    ):
        kernels = None
        for i, desc in enumerate(kernels_desc):
            weight = desc['weight']
            features = []
            for modality, sigma in desc.items():
                if modality == 'weight':
                    continue
                if modality == 'xy':
                    feature = AffinityLoss._get_mesh(
                        N, height_pred, width_pred, device)
                else:
                    # assert modality in sample, 'Modality {} is listed in {}-th kernel descriptor, but not present in the sample'.format(modality, i)
                    feature = sample
                    feature = AffinityLoss._downsample(
                        feature, height_pred, width_pred
                    )
                feature /= sigma
                features.append(feature)
            features = torch.cat(features, dim=1)
            kernel = weight * \
                AffinityLoss._create_kernels_from_features(
                    features, kernels_radius)
            kernels = kernel if kernels is None else kernel + kernels
        return kernels

    @staticmethod
    def _create_kernels_from_features(features, radius):
        assert features.dim() == 4, 'Features must be a NCHW batch'
        N, C, H, W = features.shape
        kernels = AffinityLoss._unfold(features, radius)
        kernels = kernels - kernels[:, :, radius,
                                    radius, :, :].view(N, C, 1, 1, H, W)
        kernels = (-0.5 * kernels ** 2).sum(dim=1, keepdim=True).exp()
        kernels[:, :, radius, radius, :, :] = 0
        return kernels

    @staticmethod
    def _get_mesh(N, H, W, device):
        return torch.cat((
            torch.arange(0, W, 1, dtype=torch.float32, device=device).view(
                1, 1, 1, W).repeat(N, 1, H, 1),
            torch.arange(0, H, 1, dtype=torch.float32, device=device).view(
                1, 1, H, 1).repeat(N, 1, 1, W)
        ), 1)

    @staticmethod
    def _unfold(img, radius):
        assert img.dim() == 4, 'Unfolding requires NCHW batch'
        N, C, H, W = img.shape
        diameter = 2 * radius + 1
        return F.unfold(img, diameter, 1, radius).view(N, C, diameter, diameter, H, W)

    @staticmethod
    def _batch_prototype_generator(feature, sup_label, num_classes):
        N, C, H, W = feature.shape #N,C,H,W
        if len(sup_label.shape) != 4:
            sup_label = sup_label.unsqueeze(1)#N,1,H,W
        sup_label = sup_label.long()
        labeled_num = torch.zeros((num_classes, 1, 1)).to(feature.device)
        
        prototype = torch.zeros((num_classes, N, C, H, W)).to(feature.device)
        # print("prototype.shape = ",prototype.shape)
        for num in range(num_classes):
            weight = torch.zeros_like(sup_label).to(feature.device)
            weight[sup_label==num]=1
            prototype[num][:] = feature * weight
            labeled_num[num] = weight.sum()

        # print("prototype.shape = ",prototype.shape)
        # # Unify prototype num 1 C
        labeled_num = torch.clamp(labeled_num, min=1)
        prototype = torch.sum(prototype, dim=[3,4], keepdim=False)
        prototype = torch.sum(prototype, dim=1, keepdim=True)/labeled_num
        return prototype
            
    @staticmethod
    def _persample_prototype_generator(feature, sup_label, num_classes):
        N, C, H, W = feature.shape #N,C,H,W
        # sup_label = sup_label.unsqueeze(1)#N,1,H,W
        if len(sup_label.shape) != 4:
            sup_label = sup_label.unsqueeze(1)
        sup_label = sup_label.long()
        labeled_num = torch.zeros((num_classes, N, 1)).to(feature.device)
        
        prototype = torch.zeros((num_classes, N, C, H, W)).to(feature.device)
        # print("prototype.shape = ",prototype.shape)
        for num in range(num_classes):
            weight = torch.zeros_like(sup_label).to(feature.device)
            weight[sup_label==num]=1
            prototype[num][:] = feature * weight
            labeled_num[num] = torch.sum(weight, dim=[2,3], keepdim=False)

        print("prototype.shape = ",prototype.shape)
        # # Unify prototype num N C
        labeled_num = torch.clamp(labeled_num, min=1)
        prototype = torch.sum(prototype, dim=[3,4], keepdim=False)/labeled_num
        return prototype
            
    @staticmethod
    def _affinity(feature, pro_vector, num_classes):
        # print("pro_vector.shape = ",pro_vector.shape)
        N, C, H, W = feature.shape
        # Reshape feature to (N, H, W, C)&pro_vector to (num_class, N, C)
        feature = feature.transpose(1,3).transpose(1,2)
        feature = F.normalize(feature, p=2, dim=3)#N,H,W,C
        # pro_vector = pro_vector.view(N, C, num_classes).transpose(0,2).transpose(1,2)
        # print("pro_vector.shape = ",pro_vector.shape)
        # print(feature.shape)
        cosine_similarities_prototype = torch.zeros((N, H, W, num_classes)).to(feature.device)
 
        
        for c in range(num_classes):
            pix_prototype = pro_vector[c,:,:].unsqueeze(1).unsqueeze(1)#N,1,1,C
            # pix_prototype = F.normalize(pix_prototype, p=2, dim=-1)
            # print("pix_prototype.shape=",pix_prototype.shape)
            cosine_similarities_prototype[:,:,:,c] = F.cosine_similarity(feature,pix_prototype,dim=-1)
        cosine_similarities_prototype[cosine_similarities_prototype<0] = 0

        ###just max num_classes
        # max_values, _ = torch.max(cosine_similarities_prototype, dim=-1, keepdim=True)
        # result = torch.zeros_like(cosine_similarities_prototype).to(feature.device)
        # result[max_values == cosine_similarities_prototype] = cosine_similarities_prototype[max_values == cosine_similarities_prototype]
        ###

        affinity_map = cosine_similarities_prototype/torch.clamp(torch.sum(cosine_similarities_prototype, dim=-1, keepdim=True), min=1e-10)
        affinity_map = affinity_map.transpose(1,3).transpose(2,3)
        # print("affinity.shape = ", affinity_map.shape)
        return affinity_map# N,num_class,H,W



class AffinityLoss_PixelsAssignment(torch.nn.Module):
    def forward(
            self, encoder_feature, decoder_feature, sup_label, y_hat, kernels_desc, kernels_radius, sample, height_input, width_input,Unlabeled_RoIs,Unlabeled_RoIs_sup,num_classes
    ):
        """
        Performs the forward pass of the loss.
        :param y_hat: A tensor of predicted per-pixel class probabilities of size NxCxHxW
        :param kernels_desc: A list of dictionaries, each describing one Gaussian kernel composition from modalities.
            The final kernel is a weighted sum of individual kernels. Following example is a composition of
            RGBXY and XY kernels:
            kernels_desc: [{
                'weight': 0.9,          # Weight of RGBXY kernel
                'xy': 6,                # Sigma for XY
                'rgb': 0.1,             # Sigma for RGB
            },{
                'weight': 0.1,          # Weight of XY kernel
                'xy': 6,                # Sigma for XY
            }]
        :param kernels_radius: Defines size of bounding box region around each pixel in which the kernel is constructed.
        :param sample: A dictionary with modalities (except 'xy') used in kernels_desc parameter. Each of the provided
            modalities is allowed to be larger than the shape of y_hat_softmax, in such case downsampling will be
            invoked. Default downsampling method is area resize; this can be overriden by setting.
            custom_modality_downsamplers parameter.
        :param width_input, height_input: Dimensions of the full scale resolution of modalities
        :return: Loss function value.
        """

         # refined_prediction
        N, C, height_pred, width_pred = y_hat.shape
        device = y_hat.device
        _, _, height_encoder_feature, width_encoder_feature = encoder_feature.shape
        _, _, height_decoder_feature, width_decoder_feature = decoder_feature.shape
    # ###encoder prototype & feature map
        
        sup_label_encoder = F.interpolate(sup_label.unsqueeze(1).float(), size=(height_encoder_feature, width_encoder_feature), mode='nearest')
        sup_label_decoder = F.interpolate(sup_label.unsqueeze(1).float(), size=(height_decoder_feature, width_decoder_feature), mode='nearest')
        # print("sup_label_downsample.shape = ", sup_label_downsample.shape)
        batch_prototype = AffinityLoss._batch_prototype_generator(encoder_feature, sup_label_encoder, num_classes).repeat(1,N,1)#num,N,C1
        persample_prototype = AffinityLoss._persample_prototype_generator(decoder_feature, sup_label_decoder, num_classes)#num,N,C2
        prototype = torch.cat((batch_prototype, persample_prototype), dim=-1)#num,N,(C1+C2)
 
        feature_encoder_upsample = F.interpolate(encoder_feature, size=(height_pred, width_pred), mode='bilinear', align_corners=False)#N,C1,H,W
        feature_decoder_upsample = F.interpolate(decoder_feature, size=(height_pred, width_pred), mode='bilinear', align_corners=False)#N,C2,H,W
        feature_upsample = torch.cat((feature_encoder_upsample, feature_decoder_upsample), dim=1)#N,(C1+C2),H,W
        prototype_l2 = F.normalize(prototype, p=2, dim=2)
        affinity_map = AffinityLoss._affinity(feature_upsample, prototype_l2, num_classes)#
        # ## redefined prediction(Affinity)
        y_redefined = affinity_map * y_hat
        y_hat_softmax=torch.softmax(y_redefined, dim=1)
        assert y_hat_softmax.dim() == 4, 'Prediction must be a NCHW batch'
        ###

        # ## bounder
        
        bounder_loss = AffinityLoss_PixelsAssignment._bounder_aware_loss(y_redefined, y_hat, Unlabeled_RoIs)
        

        assert width_input % width_pred == 0 and height_input % height_pred == 0 and \
            width_input * height_pred == height_input * width_pred, \
            f'[{width_input}x{height_input}] !~= [{width_pred}x{height_pred}]'

        kernels = self._create_kernels(
            kernels_desc, kernels_radius, sample, N, height_pred, width_pred, device
        )
        # kernels = kernels * (Unlabeled_RoIs_sup.unsqueeze(2).unsqueeze(2))

        # denom = N * height_pred * width_pred
        denom = Unlabeled_RoIs_sup.sum()

        y_hat_unfolded = self._unfold(y_hat_softmax, kernels_radius)
        # kernels = kernels * (Unlabeled_RoIs.unsqueeze(2).unsqueeze(2))
        # print(kernels.shape)
        product_kernel_x_y_hat = (kernels * y_hat_unfolded) \
            .view(N, C, (kernels_radius * 2 + 1) ** 2, height_pred, width_pred) \
            .sum(dim=2, keepdim=False)


        # Using shortcut for Pott's class compatibility model
        loss = -(product_kernel_x_y_hat * y_hat_softmax * Unlabeled_RoIs_sup).sum()
        # comment out to save computation, total loss may go below 0
        loss = kernels.sum() + loss
        loss = torch.clamp(loss, min=1e-5)


        out = {
            'loss': loss / denom,'bounder_loss':bounder_loss, 'prediction_redefined':y_redefined, 'heated_map':product_kernel_x_y_hat
        }

        return out

    @staticmethod
    def _downsample(img, height_dst, width_dst):
        f_down = F.adaptive_avg_pool2d
        return f_down(img, (height_dst, width_dst))

    @staticmethod
    def _create_kernels(
            kernels_desc, kernels_radius, sample, N, height_pred, width_pred, device
    ):
        kernels = None
        for i, desc in enumerate(kernels_desc):
            weight = desc['weight']
            features = []
            for modality, sigma in desc.items():
                if modality == 'weight':
                    continue
                if modality == 'xy':
                    feature = AffinityLoss_PixelsAssignment._get_mesh(
                        N, height_pred, width_pred, device)
                else:
                    # assert modality in sample, 'Modality {} is listed in {}-th kernel descriptor, but not present in the sample'.format(modality, i)
                    feature = sample
                    feature = AffinityLoss_PixelsAssignment._downsample(
                        feature, height_pred, width_pred
                    )
                feature /= sigma
                features.append(feature)
            features = torch.cat(features, dim=1)
            kernel = weight * \
                AffinityLoss_PixelsAssignment._create_kernels_from_features(
                    features, kernels_radius)
            kernels = kernel if kernels is None else kernel + kernels
        return kernels

    @staticmethod
    def _create_kernels_from_features(features, radius):
        assert features.dim() == 4, 'Features must be a NCHW batch'
        N, C, H, W = features.shape
        kernels = AffinityLoss_PixelsAssignment._unfold(features, radius)
        kernels = kernels - kernels[:, :, radius,
                                    radius, :, :].view(N, C, 1, 1, H, W)
        kernels = (-0.5 * kernels ** 2).sum(dim=1, keepdim=True).exp()
        kernels[:, :, radius, radius, :, :] = 0
        return kernels

    @staticmethod
    def _get_mesh(N, H, W, device):
        return torch.cat((
            torch.arange(0, W, 1, dtype=torch.float32, device=device).view(
                1, 1, 1, W).repeat(N, 1, H, 1),
            torch.arange(0, H, 1, dtype=torch.float32, device=device).view(
                1, 1, H, 1).repeat(N, 1, 1, W)
        ), 1)

    @staticmethod
    def _unfold(img, radius):
        assert img.dim() == 4, 'Unfolding requires NCHW batch'
        N, C, H, W = img.shape
        diameter = 2 * radius + 1
        return F.unfold(img, diameter, 1, radius).view(N, C, diameter, diameter, H, W)

    @staticmethod
    def _batch_prototype_generator(feature, sup_label, num_classes):
        N, C, H, W = feature.shape #N,C,H,W
        if len(sup_label.shape) != 4:
            sup_label = sup_label.unsqueeze(1)#N,1,H,W
        sup_label = sup_label.long()
        labeled_num = torch.zeros((num_classes, 1, 1)).to(feature.device)
        
        prototype = torch.zeros((num_classes, N, C, H, W)).to(feature.device)
        # print("prototype.shape = ",prototype.shape)
        for num in range(num_classes):
            weight = torch.zeros_like(sup_label).to(feature.device)
            weight[sup_label==num]=1
            prototype[num][:] = feature * weight
            labeled_num[num] = weight.sum()

        # print("prototype.shape = ",prototype.shape)
        # # Unify prototype num N C
        labeled_num = torch.clamp(labeled_num, min=1)
        prototype = torch.sum(prototype, dim=[3,4], keepdim=False)
        prototype = torch.sum(prototype, dim=1, keepdim=True)/labeled_num
        return prototype
            
    @staticmethod
    def _persample_prototype_generator(feature, sup_label, num_classes):
        N, C, H, W = feature.shape #N,C,H,W
        if len(sup_label.shape) != 4:
            sup_label = sup_label.unsqueeze(1)
        sup_label = sup_label.long()
        labeled_num = torch.zeros((num_classes, N, 1)).to(feature.device)
        
        prototype = torch.zeros((num_classes, N, C, H, W)).to(feature.device)
        # print("prototype.shape = ",prototype.shape)
        for num in range(num_classes):
            weight = torch.zeros_like(sup_label).to(feature.device)
            weight[sup_label==num]=1
            prototype[num][:] = feature * weight
            labeled_num[num] = torch.sum(weight, dim=[2,3], keepdim=False)

        print("prototype.shape = ",prototype.shape)
        # # Unify prototype num N C
        labeled_num = torch.clamp(labeled_num, min=1)
        prototype = torch.sum(prototype, dim=[3,4], keepdim=False)/labeled_num
        return prototype
            
    @staticmethod
    def _affinity(feature, pro_vector, num_classes):
        # print("pro_vector.shape = ",pro_vector.shape)
        N, C, H, W = feature.shape
        # Reshape feature to (N, H, W, C)&pro_vector to (num_class, N, C)
        feature = feature.transpose(1,3).transpose(1,2)
        feature = F.normalize(feature, p=2, dim=3)#N,H,W,C
        # pro_vector = pro_vector.view(N, C, num_classes).transpose(0,2).transpose(1,2)
        # print("pro_vector.shape = ",pro_vector.shape)
        # print(feature.shape)
        cosine_similarities_prototype = torch.zeros((N, H, W, num_classes)).to(feature.device)
 
        
        for c in range(num_classes):
            pix_prototype = pro_vector[c,:,:].unsqueeze(1).unsqueeze(1)#N,1,1,C
            # pix_prototype = F.normalize(pix_prototype, p=2, dim=-1)
            # print("pix_prototype.shape=",pix_prototype.shape)
            cosine_similarities_prototype[:,:,:,c] = F.cosine_similarity(feature,pix_prototype,dim=-1)
        cosine_similarities_prototype[cosine_similarities_prototype<0] = 0

        ###just max num_classes
        # max_values, _ = torch.max(cosine_similarities_prototype, dim=-1, keepdim=True)
        # result = torch.zeros_like(cosine_similarities_prototype).to(feature.device)
        # result[max_values == cosine_similarities_prototype] = cosine_similarities_prototype[max_values == cosine_similarities_prototype]
        ###

        affinity_map = cosine_similarities_prototype/torch.clamp(torch.sum(cosine_similarities_prototype, dim=-1, keepdim=True), min=1e-10)
        affinity_map = affinity_map.transpose(1,3).transpose(2,3)
        # print("affinity.shape = ", affinity_map.shape)
        return affinity_map# N,num_class,H,W
    @staticmethod
    def _bounder_aware_loss(soft_label, y_hat, Unlabeled_RoIs):
        y_hat_softmax=torch.softmax(y_hat, dim=1)
        soft_label = soft_label.transpose(1,3).transpose(1,2).detach()# N,H,W,num_class
        # ##max
        max_values, _ = torch.max(soft_label, dim=-1, keepdim=True)
        result = torch.zeros_like(soft_label).to(y_hat.device)
        result[max_values == soft_label] = soft_label[max_values == soft_label]
        soft_label = torch.softmax(result, dim=-1)

        # soft_label = torch.softmax(soft_label, dim=-1)
        soft_label = soft_label.transpose(1,3).transpose(2,3)# N,num_class,H,W
        # soft_label = soft_label.transpose(1,3).transpose(2,3)# N,num_class,H,W
        soft_label_bounder = soft_label * Unlabeled_RoIs
        y_hat_softmax_bounder = y_hat_softmax * Unlabeled_RoIs
        return losses.dice_loss1(y_hat_softmax_bounder, soft_label_bounder)






