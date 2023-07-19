import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, variances, total_labels, nms_threshold=0.5, max_total_size=200, score_threshold=0.5):
        super(Decoder, self).__init__()
        self.variances = variances
        self.total_labels = total_labels
        self.nms_threshold = nms_threshold
        self.max_total_size = max_total_size
        self.score_threshold = score_threshold

    def forward(self, roi_bboxes, pred_deltas, pred_label_probs):
        batch_size = pred_deltas.size(0)

        pred_deltas = pred_deltas.view(batch_size, -1, self.total_labels, 4)
        pred_deltas *= self.variances

        expanded_roi_bboxes = roi_bboxes.unsqueeze(-2).expand(-1, -1, self.total_labels, -1)
        pred_bboxes = bbox_utils.get_bboxes_from_deltas(expanded_roi_bboxes, pred_deltas)

        pred_labels_map = pred_label_probs.argmax(-1).unsqueeze(-1)
        pred_labels = torch.where(pred_labels_map != 0, pred_label_probs, torch.zeros_like(pred_label_probs))

        final_bboxes, final_scores, final_labels, valid_detections = bbox_utils.non_max_suppression(
            pred_bboxes, pred_labels,
            iou_threshold=self.nms_threshold,
            max_output_size_per_class=self.max_total_size,
            max_total_size=self.max_total_size,
            score_threshold=self.score_threshold
        )

        # If there are any valid detection -> Apply NMS but without score threshold
        no_detections = valid_detections[0] == 0
        if no_detections:
            final_bboxes, final_scores, final_labels, valid_detections = bbox_utils.non_max_suppression(
                pred_bboxes, pred_labels,
                iou_threshold=self.nms_threshold,
                max_output_size_per_class=self.max_total_size,
                max_total_size=self.max_total_size
            )

        # Take only valid outputs, remove zero padding -> only valid for batchsize=1
        if batch_size == 1:
            final_bboxes = final_bboxes[0, :valid_detections[0], :]
            final_scores = final_scores[0, :valid_detections[0]]
            final_labels = final_labels[0, :valid_detections[0]]

        if no_detections:
            best_score = final_scores.max(dim=1)[0]
            if best_score < 0.001:  # no good bbox
                final_bboxes = torch.zeros((1, 1, 4))
                final_labels = torch.zeros((1, 1))
                final_scores = torch.zeros((1, 1))
            else:
                better_detection_index = final_scores.argmax(dim=1)
                final_bboxes = final_bboxes[:, better_detection_index, :]
                final_scores = final_scores[:, better_detection_index]
                final_labels = final_labels[:, better_detection_index]

        return final_bboxes, final_labels, final_scores
    

class ProposalLayer(nn.Module):
    def __init__(self, base_anchors, mode, cfg):
        super(ProposalLayer, self).__init__()
        self.base_anchors = base_anchors
        self.cfg = cfg
        self.mode = mode

    def forward(self, inputs):
        rpn_bbox_deltas = inputs[0]
        rpn_labels = inputs[1]
        anchors = bbox_utils.generate_anchors((rpn_labels.size(1), rpn_labels.size(2)), self.base_anchors)

        pre_nms_topn = self.cfg.PRE_NMS_TOPN if self.mode == "training" else self.cfg.TEST_PRE_NMS_TOPN
        post_nms_topn = self.cfg.TRAIN_NMS_TOPN if self.mode == "training" else self.cfg.TEST_NMS_TOPN
        nms_iou_threshold = self.cfg.NMS_IOU_THRESHOLD
        variances = self.cfg.VARIANCES
        total_anchors = anchors.size(0)
        batch_size = rpn_bbox_deltas.size(0)
        rpn_bbox_deltas = rpn_bbox_deltas.view(batch_size, total_anchors, 4)
        rpn_labels = rpn_labels.view(batch_size, total_anchors)

        rpn_bbox_deltas *= variances
        rpn_bboxes = bbox_utils.get_bboxes_from_deltas(anchors, rpn_bbox_deltas)

        # if there are fewer possible anchors than pre nms, then take all of them
        if rpn_labels.size(1) < pre_nms_topn:
            pre_nms_topn = rpn_labels.size(1)

        _, pre_indices = rpn_labels.topk(pre_nms_topn, dim=1)

        # take top rois and apply NMS
        pre_roi_bboxes = rpn_bboxes.gather(1, pre_indices.unsqueeze(2).expand(-1, -1, 4))
        pre_roi_labels = rpn_labels.gather(1, pre_indices.unsqueeze(2))

        pre_roi_bboxes = pre_roi_bboxes.view(batch_size, pre_nms_topn, 1, 4)
        pre_roi_labels = pre_roi_labels.view(batch_size, pre_nms_topn, 1)

        roi_bboxes, _, _, _ = bbox_utils.non_max_suppression(
            pre_roi_bboxes, pre_roi_labels,
            max_output_size_per_class=post_nms_topn,
            max_total_size=post_nms_topn,
            iou_threshold=nms_iou_threshold
        )
        return roi_bboxes.detach()

import torch
import torch.nn as nn
import torch.nn.functional as F

class ProposalTargetLayer(nn.Module):
    def __init__(self, cfg, img_height, img_width):
        super(ProposalTargetLayer, self).__init__()
        self.cfg = cfg
        self.img_height = img_height
        self.img_width = img_width

    def forward(self, inputs):
        img_shape = inputs[0]
        roi_bboxes = inputs[1]
        gt_boxes = inputs[2]
        gt_labels = inputs[3]
        if self.cfg.MASK_REG:
            gt_masks = inputs[4]

        total_labels = self.cfg.NUM_CLASSES
        variances = self.cfg.VARIANCES

        # Calculate iou values between each bboxes and ground truth boxes
        iou_map, _ = bbox_utils.generate_iou_map(roi_bboxes, gt_boxes)
        # Get max index value for each row
        max_indices_each_gt_box = torch.argmax(iou_map, dim=2)
        # IoU map has iou values for every gt boxes and we merge these values column wise
        merged_iou_map = torch.max(iou_map, dim=2)

        # select positive and negative rois according to the thresholds
        pos_mask = merged_iou_map > self.cfg.TRAIN_FG_THRES
        neg_mask = (merged_iou_map < self.cfg.TRAIN_BG_THRESH_HI) & (merged_iou_map > self.cfg.TRAIN_BG_THRESH_LO)

        # Calculate positive and negative total number of rois
        positive_count = torch.sum(pos_mask, dim=1)
        max_pos_bboxes = torch.round(self.cfg.TRAIN_ROIS_PER_IMAGE * self.cfg.ROI_POSITIVE_RATIO).to(torch.int32)
        total_pos_bboxes = torch.minimum(max_pos_bboxes, positive_count)
        negative_count = torch.sum(neg_mask, dim=1)
        negative_max2 = self.cfg.TRAIN_ROIS_PER_IMAGE - total_pos_bboxes
        total_neg_bboxes = torch.minimum(negative_max2, negative_count)
        positive_count = total_pos_bboxes[0]
        negative_count = total_neg_bboxes[0]

        # Take random positive and negative rois without replacement
        if positive_count > 0:
            pos_mask = train_utils.randomly_select_xyz_mask(pos_mask, total_pos_bboxes)
        if negative_count > 0:
            neg_mask = train_utils.randomly_select_xyz_mask(neg_mask, total_neg_bboxes)

        # take corresponding gt boxes and gt labels to rois
        gt_boxes_map = gt_boxes.gather(1, max_indices_each_gt_box.unsqueeze(2))
        expanded_gt_boxes = torch.where(pos_mask.unsqueeze(-1), gt_boxes_map, torch.zeros_like(gt_boxes_map))

        gt_labels_map = gt_labels.gather(1, max_indices_each_gt_box.unsqueeze(2))
        pos_gt_labels = torch.where(pos_mask, gt_labels_map, torch.full_like(gt_labels_map, -1, dtype=torch.int32))
        neg_gt_labels = neg_mask.to(torch.int32)
        expanded_gt_labels = (pos_gt_labels + neg_gt_labels).to(torch.int32)  # (batch_size, num_rois, 4)

        # take positive gt bboxes, labels and rois
        pos_indices = pos_mask.nonzero(as_tuple=True)
        positive_count = pos_indices[0].size(0)
        gt_boxes_pos = expanded_gt_boxes[pos_indices]
        positive_rois = roi_bboxes[pos_indices]
        pos_gt_labels = expanded_gt_labels[pos_indices]

        # take negative gt bboxes, labels and rois
        neg_indices = neg_mask.nonzero(as_tuple=True)
        gt_boxes_neg = expanded_gt_boxes[neg_indices]
        neg_rois = roi_bboxes[neg_indices]
        neg_gt_labels = expanded_gt_labels[neg_indices]

        # concat positive + negative gt bboxes, labels and rois
        total_gt_bboxes = torch.cat([gt_boxes_pos, gt_boxes_neg], 0)
        total_gt_labels = torch.cat([pos_gt_labels, neg_gt_labels], 0)
        total_rois = torch.cat([positive_rois, neg_rois], 0)

        # get deltas from bboxes
        gt_bbox_deltas = bbox_utils.get_deltas_from_bboxes(total_rois, total_gt_bboxes) / variances
        gt_bbox_labels = total_gt_labels

        # Transform to one hot representation (batch_size, num_rois, num_classes)
        gt_bbox_labels = F.one_hot(gt_bbox_labels, total_labels)

        gt_bbox_deltas = gt_bbox_deltas.unsqueeze(0)
        gt_bbox_labels = gt_bbox_labels.unsqueeze(0)
        total_rois = total_rois.unsqueeze(0)

        if self.cfg.MASK_REG:
            # Take only positive rois for mask training and corresponding roi_gt_boxes
            roi_gt_boxes = gt_boxes_map[pos_indices]

            y1, x1, y2, x2 = torch.split(positive_rois, 1, dim=1)
            y1t, x1t, y2t, x2t = torch.split(roi_gt_boxes, 1, dim=1)

            # compute overlap between roi coordinate and gt_roi coordinate
            x1o = torch.max(x1, x1t)
            y1o = torch.max(y1, y1t)
            x2o = torch.min(x2, x2t)
            y2o = torch.min(y2, y2t)

            if positive_count != 0:
                # Calculate labels in original mask -> gt_masks=(batch_size, num_masks, img_height, img_width)
                original_affordance_labels = torch.unique(gt_masks.view(-1))
                original_affordance_labels = torch.sort(original_affordance_labels)[0]

                # filter indices of gt boxes
                indices_pos_gt_boxes = max_indices_each_gt_box[pos_mask]

                # mask associated wrt to gt bbox (batch_size, positive_rois, mask_size, mask_size)
                gt_mask = gt_masks.gather(1, indices_pos_gt_boxes.unsqueeze(1))

                gt_mask = gt_mask.unsqueeze(4).to(torch.float32)
                y1o = y1o.squeeze()
                x1o = x1o.squeeze()
                y2o = y2o.squeeze()
                x2o = x2o.squeeze()

                # create boxes to crop and indexes where each mask has its own box
                boxes = torch.stack([y1o, x1o, y2o, x2o], dim=1).to(torch.float32)

                # remove batch dim -> needed for crop and resize op
                img_shape = img_shape.squeeze()
                gt_mask = gt_mask.squeeze()

                # crop and resize the masks individually
                positive_masks = self._crop_and_resize_masks_no_resize(img_shape, gt_mask, boxes, positive_rois,
                                                                       positive_count, original_affordance_labels)
                # Add batch dim
                positive_masks = positive_masks.unsqueeze(0)
                positive_rois = positive_rois.unsqueeze(0)
                masks = positive_masks
            else:
                positive_rois = positive_rois.unsqueeze(0)
                masks = torch.zeros(1, 0, self.cfg.TRAIN_MASK_SIZE, self.cfg.TRAIN_MASK_SIZE).to(torch.int32)

            return total_rois, gt_bbox_deltas.detach(), gt_bbox_labels.detach(), masks.detach(), \
                positive_rois.detach()

        return gt_bbox_deltas.detach(), gt_bbox_labels.detach()

    def _crop_and_resize_masks_no_resize(self, img_shape, masks, overlapping_boxes, rois, positive_count, original_aff_labels):
        # denormalize bboxes
        overlapping_boxes = bbox_utils.denormalize_bboxes(overlapping_boxes, img_shape[0], img_shape[1]).to(torch.int32)
        rois = bbox_utils.denormalize_bboxes(rois, img_shape[0], img_shape[1]).to(torch.int32)

        num_masks = masks.size(0)
        final_masks = torch.zeros(num_masks, self.cfg.TRAIN_MASK_SIZE, self.cfg.TRAIN_MASK_SIZE)
        for i in range(num_masks):
            mask = masks[i]

            # get roi and overlap area coordinates
            y1, x1, y2, x2 = rois[i].split(1, dim=0)
            y1, x1, y2, x2 = y1.squeeze(), x1.squeeze(), y2.squeeze(), x2.squeeze()

            y1o, x1o, y2o, x2o = overlapping_boxes[i].split(1, dim=0)
            y1o, x1o, y2o, x2o = y1o.squeeze(), x1o.squeeze(), y2o.squeeze(), x2o.squeeze()

            # take overlap area between gt_bbox and roi
            overlapping_mask_area = mask[y1o:y2o, x1o:x2o]

            # calculate offsets with 0 above and in the left of the overlapping area
            offset_height = y1o - y1
            offset_width = x1o - x1

            # calculate roi height and width
            target_height = y2 - y1 + 1
            target_width = x2 - x1 + 1

            roi_mask = F.pad(overlapping_mask_area, (offset_width.item(), 0, offset_height.item(), 0))

            # resize to mask size
            roi_mask = F.interpolate(roi_mask.unsqueeze(0).unsqueeze(0), size=(self.cfg.TRAIN_MASK_SIZE, self.cfg.  TRAIN_MASK_SIZE), mode='bilinear').squeeze()

            # Create a structure with 0 for all masks except for the current mask and add that to final mask structure
            temp_masks = torch.zeros(num_masks, self.cfg.TRAIN_MASK_SIZE, self.cfg.TRAIN_MASK_SIZE)
            temp_masks[i] = roi_mask.squeeze()
            final_masks += temp_masks

        final_masks = self._convert_mask_to_original_ids_manual(positive_count, final_masks, original_aff_labels, self.cfg. TRAIN_MASK_SIZE)
        return final_masks

    def _reset_mask_ids(self, mask, before_uni_ids):
        # reset ID mask values from [0, 1, 4] to [0, 1, 2] to resize later
        counter = 0
        final_mask = torch.zeros_like(mask)
        for id in before_uni_ids:
            temp_mask = torch.where(mask == id, counter, 0)
            final_mask += temp_mask
            counter += 1
        return final_mask

    def _convert_mask_to_original_ids_manual(self, positive_count, mask, original_uni_ids, train_mask_size):
        const = 0.005
        original_uni_ids_2 = original_uni_ids.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        dif = torch.abs(mask.float() - original_uni_ids_2.float()) < const
        max = torch.argmax(dif, dim=0).unsqueeze(3)
        # create mask array where each position contains the original_uni_ids
        temp_mask = torch.where(torch.zeros(positive_count, train_mask_size, train_mask_size, 1).to(torch.bool),    original_uni_ids, original_uni_ids)
        return temp_mask.gather(3, max).squeeze()
    

        return total_rois, gt_bbox_deltas.detach(), gt_bbox_labels.detach(), masks.detach(), \
            positive_rois.detach()

        return gt_bbox_deltas.detach(), gt_bbox_labels.detach()

    def _crop_and_resize_masks_no_resize(self, img_shape, masks, overlapping_boxes, rois, positive_count, original_aff_labels):
        # denormalize bboxes
        overlapping_boxes = bbox_utils.denormalize_bboxes(overlapping_boxes, img_shape[0], img_shape[1]).to(torch.int32)
        rois = bbox_utils.denormalize_bboxes(rois, img_shape[0], img_shape[1]).to(torch.int32)

        num_masks = masks.size(0)
        final_masks = torch.zeros(num_masks, self.cfg.TRAIN_MASK_SIZE, self.cfg.TRAIN_MASK_SIZE)
        for i in range(num_masks):
            mask = masks[i]

            # get roi and overlap area coordinates
            y1, x1, y2, x2 = rois[i].split(1, dim=0)
            y1, x1, y2, x2 = y1.squeeze(), x1.squeeze(), y2.squeeze(), x2.squeeze()

            y1o, x1o, y2o, x2o = overlapping_boxes[i].split(1, dim=0)
            y1o, x1o, y2o, x2o = y1o.squeeze(), x1o.squeeze(), y2o.squeeze(), x2o.squeeze()

            # take overlap area between gt_bbox and roi
            overlapping_mask_area = mask[y1o:y2o, x1o:x2o]

            # calculate offsets with 0 above and in the left of the overlapping area
            offset_height = y1o - y1
            offset_width = x1o - x1

            # calculate roi height and width
            target_height = y2 - y1 + 1
            target_width = x2 - x1 + 1

            roi_mask = F.pad(overlapping_mask_area, (offset_width.item(), 0, offset_height.item(), 0))

            # resize to mask size
            roi_mask = F.interpolate(roi_mask.unsqueeze(0).unsqueeze(0), size=(self.cfg.TRAIN_MASK_SIZE, self.cfg.  TRAIN_MASK_SIZE), mode='bilinear').squeeze()

            # Create a structure with 0 for all masks except for the current mask and add that to final mask structure
            temp_masks = torch.zeros(num_masks, self.cfg.TRAIN_MASK_SIZE, self.cfg.TRAIN_MASK_SIZE)
            temp_masks[i] = roi_mask.squeeze()
            final_masks += temp_masks

        final_masks = self._convert_mask_to_original_ids_manual(positive_count, final_masks, original_aff_labels, self.cfg. TRAIN_MASK_SIZE)
        return final_masks

    def _reset_mask_ids(self, mask, before_uni_ids):
        # reset ID mask values from [0, 1, 4] to [0, 1, 2] to resize later
        counter = 0
        final_mask = torch.zeros_like(mask)
        for id in before_uni_ids:
            temp_mask = torch.where(mask == id, counter, 0)
            final_mask += temp_mask
            counter += 1
        return final_mask

    def _convert_mask_to_original_ids_manual(self, positive_count, mask, original_uni_ids, train_mask_size):
        const = 0.005
        original_uni_ids_2 = original_uni_ids.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        dif = torch.abs(mask.float() - original_uni_ids_2.float()) < const
        max = torch.argmax(dif, dim=0).unsqueeze(3)
        # create mask array where each position contains the original_uni_ids
        temp_mask = torch.where(torch.zeros(positive_count, train_mask_size, train_mask_size, 1).to(torch.bool),    original_uni_ids, original_uni_ids)
        return temp_mask.gather(3, max).squeeze()

class RoiAlign(nn.Module):
    """Reducing all feature maps to same size.
    Firstly cropping bounding boxes from the feature maps and then resizing it to the pooling size.
    inputs:
        feature_map = (batch_size, img_output_height, img_output_width, channels)
        roi_bboxes = (batch_size, train/test_nms_topn, [y1, x1, y2, x2])

    outputs:
        final_pooling_feature_map = (batch_size, train/test_nms_topn, pooling_size[0], pooling_size[1], channels)
            pooling_size usually (7, 7)
    """

    def __init__(self, cfg):
        super(RoiAlign, self).__init__()
        self.cfg = cfg

    def forward(self, inputs):
        feature_map = inputs[0]
        roi_bboxes = inputs[1]
        pooling_size = (self.cfg.POOL_SIZE, self.cfg.POOL_SIZE)
        batch_size, total_bboxes = roi_bboxes.size(0), roi_bboxes.size(1)
        row_size = batch_size * total_bboxes

        # We need to arrange bbox indices for each batch
        pooling_bbox_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, total_bboxes).flatten()
        pooling_bboxes = roi_bboxes.view(row_size, 4)

        # Crop to bounding box size then resize to pooling size
        pooling_feature_map = ops.roi_align(
            feature_map,
            [pooling_bboxes],
            [pooling_bbox_indices],
            output_size=pooling_size
        )
        final_pooling_feature_map = pooling_feature_map.view(batch_size, total_bboxes,
                                                             pooling_feature_map.size(1), pooling_feature_map.size(2),
                                                             pooling_feature_map.size(3))
        return final_pooling_feature_map



import torch
import torch.nn as nn
from torchvision.models import detection

class RPNRegressionLoss(nn.Module):
    def __init__(self):
        super(RPNRegressionLoss, self).__init__()
    
    def forward(self, rpn_reg_actuals, rpn_reg_predictions, rpn_cls_actuals):
        # Implement the RPN regression loss calculation
        # rpn_reg_actuals: (batch_size, num_anchors, 4)
        # rpn_reg_predictions: (batch_size, num_anchors, 4)
        # rpn_cls_actuals: (batch_size, num_anchors)
        loss = ...
        return loss

class ClassificationLoss(nn.Module):
    def __init__(self):
        super(ClassificationLoss, self).__init__()
    
    def forward(self, frcnn_cls_actuals, frcnn_cls_predictions):
        # Implement the classification loss calculation
        # frcnn_cls_actuals: (batch_size, num_rois)
        # frcnn_cls_predictions: (batch_size, num_rois, num_classes)
        loss = ...
        return loss

class MaskLoss(nn.Module):
    def __init__(self):
        super(MaskLoss, self).__init__()
    
    def forward(self, target_masks, mask_prob_output):
        # Implement the mask loss calculation
        # target_masks: (batch_size, num_rois, mask_size, mask_size)
        # mask_prob_output: (batch_size, num_rois, num_affordance_classes, mask_size, mask_size)
        loss = ...
        return loss

class GetAffordanceNetModel(nn.Module):
    def __init__(self, feature_extractor, rpn_model, cfg, base_anchors, mode="training"):
        super(GetAffordanceNetModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.rpn_model = rpn_model
        self.cfg = cfg
        self.base_anchors = base_anchors
        self.mode = mode
        
        self.proposal_layer = ProposalLayer(base_anchors, mode, cfg)
        self.proposal_target_layer = ProposalTargetLayer(cfg, feature_extractor.output_shape[1], feature_extractor.output_shape[2])
        self.roi_align = RoiAlign(cfg)
        
        # Define other layers and losses as needed
        
    def forward(self, input_img, input_img_shape=None, input_gt_boxes=None, input_gt_labels=None,
                rpn_reg_actuals=None, rpn_cls_actuals=None, input_gt_masks=None, input_gt_seg_mask_inds=None):
        
        rpn_reg_predictions, rpn_cls_predictions = self.rpn_model(input_img)
        roi_bboxes = self.proposal_layer([rpn_reg_predictions, rpn_cls_predictions])
        
        if self.mode == "training":
            rois, frcnn_reg_actuals, frcnn_cls_actuals, target_masks, rois_pos = self.proposal_target_layer(
                [input_img_shape, roi_bboxes, input_gt_boxes, input_gt_labels, input_gt_masks, input_gt_seg_mask_inds]
            )
            roi_pooled = self.roi_align([self.feature_extractor(input_img), rois])
            
            # Implement the rest of the training process and loss calculations
            # ...
            
            return rpn_reg_loss, rpn_cls_loss, reg_loss, cls_loss, mask_loss
            
        else:
            roi_pooled = self.roi_align([self.feature_extractor(input_img), roi_bboxes])
            
            # Implement the inference process
            # ...
            
            return bboxes, labels, scores, mask_prob_output


F

def init_model(model, cfg):
    """Generating dummy data for initialize model.
    In this way, the training process can continue from where it left off.
    inputs:
        model = torch.nn.Module
        cfg = configuration dictionary
    """
    final_height, final_width = cfg.IMG_SIZE_HEIGHT, cfg.IMG_SIZE_WIDTH
    batch_size = cfg.BATCH_SIZE
    num_classes = cfg.NUM_CLASSES
    num_anchors = cfg.ANCHOR_COUNT * cfg.FEATURE_MAP_SHAPE * cfg.FEATURE_MAP_SHAPE
    
    img = torch.rand((batch_size, 3, final_height, final_width))
    gt_boxes = torch.rand((batch_size, 1, 4))
    gt_labels = torch.randint(0, num_classes, (batch_size, 1))
    bbox_deltas = torch.rand((batch_size, num_anchors, 4))
    bbox_labels = torch.rand((batch_size, cfg.FEATURE_MAP_SHAPE, cfg.FEATURE_MAP_SHAPE, cfg.ANCHOR_COUNT))
    
    if cfg.MASK_REG:
        mask = torch.rand((batch_size, 1, final_height, final_width))
        mask_ids = torch.randint(0, cfg.NUM_AFFORDANCE_CLASSES, (batch_size, 1))
        model(img, gt_boxes, gt_labels, bbox_deltas, bbox_labels, mask, mask_ids)
    else:
        model(img, gt_boxes, gt_labels, bbox_deltas, bbox_labels)

def init_model_no_resize(model, cfg):
    """Generating dummy data for initialize model.
    In this way, the training process can continue from where it left off.
    inputs:
        model = torch.nn.Module
        cfg = configuration dictionary
    """
    final_height, final_width = cfg.IMG_SIZE_HEIGHT, cfg.IMG_SIZE_WIDTH
    batch_size = cfg.BATCH_SIZE
    num_classes = cfg.NUM_CLASSES
    num_anchors = cfg.ANCHOR_COUNT * cfg.FEATURE_MAP_SHAPE * cfg.FEATURE_MAP_SHAPE
    
    img = torch.rand((batch_size, 3, final_height, final_width))
    img_shape = torch.tensor([[final_height, final_width]], dtype=torch.float32)
    gt_boxes = torch.rand((batch_size, 1, 4))
    gt_labels = torch.randint(0, num_classes, (batch_size, 1))
    bbox_deltas = torch.rand((batch_size, num_anchors, 4))
    bbox_labels = torch.rand((batch_size, cfg.FEATURE_MAP_SHAPE, cfg.FEATURE_MAP_SHAPE, cfg.ANCHOR_COUNT))
    
    if cfg.MASK_REG:
        mask = torch.rand((batch_size, 1, final_height, final_width))
        mask_ids = torch.randint(0, cfg.NUM_AFFORDANCE_CLASSES, (batch_size, 1))
        model(img, img_shape, gt_boxes, gt_labels, bbox_deltas, bbox_labels, mask, mask_ids)
    else:
        model(img, img_shape, gt_boxes, gt_labels, bbox_deltas, bbox_labels)
