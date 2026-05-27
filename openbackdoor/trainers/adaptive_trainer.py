from openbackdoor.victims import Victim
from openbackdoor.utils import evaluate_classification
from openbackdoor.trainers.trainer import Trainer
from openbackdoor.data import get_dataloader, wrap_dataset
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
# from transformers.optimization import AdamW
import torch
from datetime import datetime
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from typing import *
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from umap import UMAP
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from utils.logger import get_logger
import settings
from utils.path import get_victim_save_path


class AdaptiveTrainer(Trainer):
    def __init__(self, act_lambd, rep_lambd, **kwargs):
        self.rep_lambd = rep_lambd
        self.act_lambd = act_lambd
        super().__init__(**kwargs)

        self.train_configs.update({"act_lambd": act_lambd, "rep_lambd": rep_lambd})
        # print(self.train_configs)


        # save_path = kwargs["save_path"]
        # dataset_name = kwargs["dataset_name"]
        # poison_setting = kwargs["poison_setting"]
        # poison_method = kwargs["poison_method"]
        # poison_rate = kwargs["poison_rate"]
        # name = "adaptive"

        # self.save_path = get_victim_save_path(name, save_path, dataset_name, poison_setting, poison_method, poison_rate, lambd=adaptive_lambd)


    def split_batch(self, batch):
        clean_batch = {
            key: [item for item, pl in zip(batch[key], batch["poison_label"]) if pl == 0]
            for key in batch.keys()
        }
        poison_batch = {
            key: [item for item, pl in zip(batch[key], batch["poison_label"]) if pl == 1]
            for key in batch.keys()
        }
        return clean_batch, poison_batch

    def get_activations_tensor(self, victim, inputs):
        """ Extracts [CLS] activations for all layers (except embedding layer)
        Returns a flattened tensor of shape [batch_size, num_layers * hidden_size]
        """
        outputs = getattr(victim.plm, victim.plm.base_model_prefix)(**inputs, output_hidden_states=True)
        
        # outputs.hidden_states is a tuple of (embed_layer, layer1, layer2, ...)
        # We take [:, 0, :] (the CLS token) for each layer starting from index 1
        cls_activations = [h[:, 0, :] for h in outputs.hidden_states[1:]]
        
        # Concatenate along the hidden dimension: [batch_size, layers * hidden_size]
        combined_activations = torch.cat(cls_activations, dim=-1)
        return combined_activations

    def get_representation_tensor(self, victim, inputs):
        """ Extracts [CLS] activations for the final layer only
        Returns a tensor of shape [batch_size, hidden_size]
        """
        outputs = getattr(victim.plm, victim.plm.base_model_prefix)(**inputs, output_hidden_states=True)
        
        # outputs.hidden_states[-1] retrieves the last layer (top-most layer)
        # [:, 0, :] extracts the [CLS] token representation for that layer
        final_layer_cls = outputs.hidden_states[-1][:, 0, :]
        
        return final_layer_cls

    def compute_act_reg_loss(self, victim, batch_inputs, poison_labels):
        # 1. Split the batch based on poison_labels
        # poison_labels should be a tensor/mask of 1s (poison) and 0s (clean)
        poison_mask = (poison_labels == 1)
        clean_mask = (poison_labels == 0)
        
        # Filter inputs (assuming dictionary of tensors)
        poison_inputs = {k: v[poison_mask] for k, v in batch_inputs.items()}
        clean_inputs = {k: v[clean_mask] for k, v in batch_inputs.items()}
        
        # Handle cases where one side might be empty in a specific batch
        if poison_inputs['input_ids'].size(0) == 0 or clean_inputs['input_ids'].size(0) == 0:
            return torch.tensor(0.0, device=victim.device, requires_grad=True)

        # 2. Get activations (r_i)
        r_backdoor = self.get_activations_tensor(victim, poison_inputs)
        r_clean = self.get_activations_tensor(victim, clean_inputs)
        
        # 3. Align sizes if the number of samples differs 
        # Usually, this loss is calculated pair-wise. If you have 1 poison and 1 clean:
        # We use F.pairwise_distance or manual norm as per your formula
        
        # If the formula implies a sum over the flattened vector elements:
        # First, ensure we have a 1-to-1 mapping or average the representations
        r_backdoor_mean = r_backdoor.mean(dim=0) # [L * d]
        r_clean_mean = r_clean.mean(dim=0)       # [L * d]
        
        # 4. Calculate L2 Norm of the difference (Euclidean Distance)
        # The formula shows the norm of the difference between elements i
        loss_reg = torch.norm(r_backdoor_mean - r_clean_mean, p=2) / torch.sqrt(torch.tensor(r_backdoor_mean.size(0), dtype=torch.float32))
        
        return loss_reg

    def compute_rep_reg_loss(self, victim, batch_inputs, poison_labels):
        # 1. Split the batch based on poison_labels
        # poison_labels should be a tensor/mask of 1s (poison) and 0s (clean)
        poison_mask = (poison_labels == 1)
        clean_mask = (poison_labels == 0)
        
        # Filter inputs (assuming dictionary of tensors)
        poison_inputs = {k: v[poison_mask] for k, v in batch_inputs.items()}
        clean_inputs = {k: v[clean_mask] for k, v in batch_inputs.items()}
        
        # Handle cases where one side might be empty in a specific batch
        if poison_inputs['input_ids'].size(0) == 0 or clean_inputs['input_ids'].size(0) == 0:
            return torch.tensor(0.0, device=victim.device, requires_grad=True)

        # 2. Get activations (r_i)
        r_backdoor = self.get_representation_tensor(victim, poison_inputs)
        r_clean = self.get_representation_tensor(victim, clean_inputs)
        
        # 3. Align sizes if the number of samples differs 
        # Usually, this loss is calculated pair-wise. If you have 1 poison and 1 clean:
        # We use F.pairwise_distance or manual norm as per your formula
        
        # If the formula implies a sum over the flattened vector elements:
        # First, ensure we have a 1-to-1 mapping or average the representations
        r_backdoor_mean = r_backdoor.mean(dim=0) # [L * d]
        r_clean_mean = r_clean.mean(dim=0)       # [L * d]
        
        # 4. Calculate L2 Norm of the difference (Euclidean Distance)
        # The formula shows the norm of the difference between elements i
        loss_reg = torch.norm(r_backdoor_mean - r_clean_mean, p=2)
        
        return loss_reg

    def train_one_epoch(self, epoch: int, epoch_iterator):
        """
        Train one epoch function.

        Args:
            epoch (:obj:`int`): current epoch.
            epoch_iterator (:obj:`torch.utils.data.DataLoader`): dataloader for training.
        
        Returns:
            :obj:`float`: average loss of the epoch.
        """
        self.model.train()
        total_loss = 0
        poison_loss_list, normal_loss_list = [], []
        for step, batch in enumerate(epoch_iterator):

            batch_inputs, batch_labels = self.model.process(batch)
            output = self.model(batch_inputs)
            logits = output.logits

            act_loss = torch.tensor(0.0).to(self.model.device)
            rep_loss = torch.tensor(0.0).to(self.model.device)
            if self.act_lambd > 0:
                act_loss = self.compute_act_reg_loss(self.model, batch_inputs, torch.tensor(batch["poison_label"]).to(self.model.device))
            if self.rep_lambd > 0:
                rep_loss = self.compute_rep_reg_loss(self.model, batch_inputs, torch.tensor(batch["poison_label"]).to(self.model.device))

            cls_loss = self.loss_function(logits, batch_labels)

            # if step % 100 == 0:
            #     self.logger.info(f"Step {step}: Classification Loss = {cls_loss.item()}, Regularization Loss = {reg_loss.item()}")

            loss = cls_loss + self.act_lambd * act_loss + self.rep_lambd * rep_loss
            self.losses.append(loss.item())
            self.cls_losses.append(cls_loss.item())
            self.act_losses.append(act_loss.item())
            self.rep_losses.append(rep_loss.item())

            if self.visualize:
                poison_labels = batch["poison_label"]
                for l, poison_label in zip(loss, poison_labels):
                    if poison_label == 1:
                        poison_loss_list.append(l.item())
                    else:
                        normal_loss_list.append(l.item())
                loss = loss.mean()

            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps
            
            loss.backward()


            if (step + 1) % self.gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                total_loss += loss.item()
                self.model.zero_grad()

        avg_loss = total_loss / len(epoch_iterator)
        avg_poison_loss = sum(poison_loss_list) / len(poison_loss_list) if self.visualize else 0
        avg_normal_loss = sum(normal_loss_list) / len(normal_loss_list) if self.visualize else 0
        
        return avg_loss, avg_poison_loss, avg_normal_loss