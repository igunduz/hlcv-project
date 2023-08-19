import pytorch_lightning as pl
from transformers import SegformerForSemanticSegmentation, SegformerForImageClassification
from datasets import load_metric
from utils.parse_batch import parse_encoded_input, parse_object_labels
import torch
from torch import nn
import numpy as np

# Referred to https://blog.roboflow.com/how-to-train-segformer-on-a-custom-dataset-with-pytorch-lightning/#create-a-dataset

class SegformerFinetuner(pl.LightningModule):
    
    def __init__(self, id2label, train_dataloader=None, val_dataloader=None, test_dataloader=None, metrics_interval=100):
        super(SegformerFinetuner, self).__init__()
        # Initialize the object classification model 
        self.object_classification_model = SegformerForImageClassification.from_pretrained("nvidia/mit-b0")
        self.id2label = id2label
        self.metrics_interval = metrics_interval
        self.train_dl = train_dataloader
        self.val_dl = val_dataloader
        self.test_dl = test_dataloader
        
        self.num_classes = len(self.id2label.keys())
        self.label2id = {v:k for k,v in self.id2label.items()}

        self.validation_step_outputs = []

        print(f"id2label: {self.id2label.keys()}")
        print(f"label2id: {self.label2id.keys()}")
        print(f"id2label: {type(self.id2label)}")
        print(f"label2id: {type(self.label2id)}")
        print(f"num_classes: {self.num_classes}")
        
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512", 
            return_dict=False, 
            num_labels=self.num_classes,
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True,
        )
        
        self.train_mean_iou = load_metric("mean_iou")
        self.val_mean_iou = load_metric("mean_iou")
        self.test_mean_iou = load_metric("mean_iou")

        n_gpu = torch.cuda.device_count()
        if n_gpu == 0:
            print("Warning: There\'s no GPU available on this machine,"
                "training will be performed on CPU.")
        self._device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
        self._device_ids = list(range(n_gpu))
        self.model.to(self._device)
        if len(self._device_ids) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self._device_ids)
        
    def forward(self, images, masks):
        affordance_logits = self.model(pixel_values=images, labels=masks)
        object_classification_logits = self.object_classification_model(images)
        return affordance_logits, object_classification_logits
    
    def training_step(self, batch, batch_nb):
        torch.cuda.empty_cache()
        
        images, masks = batch['pixel_values'], batch['labels']
        object_labels = batch['object_labels']
        affordance_logits, object_classification_logits = self(images, masks)
        
        # Compute affordance segmentation loss
        affordance_loss = nn.CrossEntropyLoss()(affordance_logits, masks)
        
        # Compute object classification loss
        object_classification_loss = nn.CrossEntropyLoss()(object_classification_logits, object_labels)
        
        # Combine the two losses
        total_loss = affordance_loss + object_classification_loss
        
        # Perform backpropagation
        total_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        upsampled_affordance_logits = nn.functional.interpolate(
            affordance_logits, 
            size=masks.shape[-2:], 
            mode="bilinear", 
            align_corners=False
        )

        predicted_affordance = upsampled_affordance_logits.argmax(dim=1)

        self.train_mean_iou.add_batch(
            predictions=predicted_affordance.detach().cpu().numpy(), 
            references=masks.detach().cpu().numpy()
        )
        if batch_nb % self.metrics_interval == 0:

            metrics_affordance = self.train_mean_iou.compute(
                num_labels=self.num_classes, 
                ignore_index=255, 
                reduce_labels=False,
            )
            
            metrics_affordance = {'affordance_mean_iou': metrics_affordance["mean_iou"], 'affordance_mean_accuracy': metrics_affordance["mean_accuracy"]}
            
            for k,v in metrics_affordance.items():
                self.log(k,v)
                
        # Calculate metrics for object classification
        predicted_object_class = object_classification_logits.argmax(dim=1)
        accuracy_object_class = torch.sum(predicted_object_class == object_labels).item() / object_labels.size(0)
        
        self.log('object_classification_accuracy', accuracy_object_class)
        
        return {'loss': total_loss}
    
    def validation_step(self, batch, batch_nb):
    # def validation_step(self, batch):
        torch.cuda.empty_cache()
        
        images, masks = batch['pixel_values'], batch['labels']
        
        outputs = self(images, masks)
        
        loss, logits = outputs[0], outputs[1]
        
        upsampled_logits = nn.functional.interpolate(
            logits, 
            size=masks.shape[-2:], 
            mode="bilinear", 
            align_corners=False
        )
        
        predicted = upsampled_logits.argmax(dim=1)
        
        self.val_mean_iou.add_batch(
            predictions=predicted.detach().cpu().numpy(), 
            references=masks.detach().cpu().numpy()
        )

        self.validation_step_outputs.append({'val_loss': loss})
        
        return({'val_loss': loss})
    
    def on_validation_epoch_end(self):
        metrics = self.val_mean_iou.compute(
              num_labels=self.num_classes, 
              ignore_index=255, 
              reduce_labels=False,
            )
        
        avg_val_loss = torch.stack([x["val_loss"] for x in self.validation_step_outputs]).mean()
        val_mean_iou = metrics["mean_iou"]
        val_mean_accuracy = metrics["mean_accuracy"]
        
        metrics = {"val_loss": avg_val_loss, "val_mean_iou":val_mean_iou, "val_mean_accuracy":val_mean_accuracy}
        for k,v in metrics.items():
            self.log(k,v)

        self.validation_step_outputs.clear()

        return metrics
    
    def test_step(self, batch, batch_nb):
        torch.cuda.empty_cache()
        
        images, masks = batch['pixel_values'], batch['labels']
        
        outputs = self(images, masks)
        
        loss, logits = outputs[0], outputs[1]
        
        upsampled_logits = nn.functional.interpolate(
            logits, 
            size=masks.shape[-2:], 
            mode="bilinear", 
            align_corners=False
        )
        
        predicted = upsampled_logits.argmax(dim=1)
        
        self.test_mean_iou.add_batch(
            predictions=predicted.detach().cpu().numpy(), 
            references=masks.detach().cpu().numpy()
        )
            
        return({'test_loss': loss})
    
    def test_epoch_end(self, outputs):
        metrics = self.test_mean_iou.compute(
              num_labels=self.num_classes, 
              ignore_index=255, 
              reduce_labels=False,
          )
       
        avg_test_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        test_mean_iou = metrics["mean_iou"]
        test_mean_accuracy = metrics["mean_accuracy"]

        metrics = {"test_loss":avg_test_loss, "test_mean_iou":test_mean_iou, "test_mean_accuracy":test_mean_accuracy}
        
        for k,v in metrics.items():
            self.log(k,v)
        
        return metrics
    
    def configure_optimizers(self):
        return torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=2e-05, eps=1e-08)
    
    def train_dataloader(self):
        return self.train_dl
    
    def val_dataloader(self):
        return self.val_dl
    
    def test_dataloader(self):
        return self.test_dl