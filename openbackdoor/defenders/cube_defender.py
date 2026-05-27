from .defender import Defender
from openbackdoor.victims import PLMVictim, Victim
from openbackdoor.data import get_dataloader, collate_fn
# from utils.logger import get_logger
from utils.logger import get_logger
from openbackdoor.trainers import Trainer
from typing import *
from torch.utils.data import DataLoader
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.decomposition import PCA
from umap import UMAP
from hdbscan import HDBSCAN
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from utils.path import get_newest_directory





class CUBEDefender(Defender):
    r"""
        Defender for `CUBE <https://arxiv.org/abs/2206.08514>`_
    
    Args:
        epochs (`int`, optional): Number of CUBE encoder training epochs. Default to 10.
        batch_size (`int`, optional): Batch size. Default to 32.
        lr (`float`, optional): Learning rate for RAP trigger embeddings. Default to 2e-5.
        num_classes (:obj:`int`, optional): The number of classes. Default to 2.
        model_name (`str`, optional): The model's name to help filter poison samples. Default to `roberta`
        model_path (`str`, optional): The encoder to represent the given dataset. Default to `roberta-base`
    """
    def __init__(
        self,
        cube_warm_up_epochs: Optional[int] = 0,
        cube_epochs: Optional[int] = 10,
        cube_batch_size: Optional[int] = 32,
        cube_lr: Optional[float] = 2e-5,
        cube_num_classes: Optional[int] = 2,
        cube_model_name: Optional[str] = 'roberta',
        cube_model_load_path: Optional[str] = '',
        cube_model_save_path: Optional[str] = '',
        cube_load_model: Optional[bool] = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pre = False
        self.warm_up_epochs = cube_warm_up_epochs
        self.epochs = cube_epochs
        self.batch_size = cube_batch_size
        self.lr = float(cube_lr)
        self.num_classes = cube_num_classes
        self.logger = get_logger(__name__)
        self.load = cube_load_model

        self.model_load_path = cube_model_load_path

        
        if cube_load_model and self.model_load_path and os.path.exists(self.model_load_path):
            self.logger.info("Loading pre-trained CUBE encoder from {}".format(self.model_load_path))
            self.encoder = PLMVictim(model=cube_model_name,num_classes=cube_num_classes, load_path=self.model_load_path, load=True)
            self.trainer = Trainer(warm_up_epochs=cube_warm_up_epochs, epochs=cube_epochs, 
                                batch_size=cube_batch_size, lr=self.lr,
                                save_path=cube_model_save_path, ckpt='last')
        else:
            self.encoder = PLMVictim(model=cube_model_name, num_classes=cube_num_classes)
            self.trainer = Trainer(warm_up_epochs=cube_warm_up_epochs, epochs=cube_epochs, 
                                batch_size=cube_batch_size, lr=self.lr,
                                save_path=cube_model_save_path, ckpt='last')
        

    def correct(
        self, 
        poison_data: List,
        clean_data: Optional[List] = None, 
        model: Optional[Victim] = None
    ):

        # Step 1. Encoding
        embeddings, y_true = self.encode(poison_data)

        # Step 2. Clustering
        y_pred = self.clustering(embeddings)

        # Step 3. Filtering
        filtered_dataset = self.filtering(poison_data, y_true, y_pred)

        return filtered_dataset

    def detect(
        self, 
        model: Optional[Victim],
        clean_data: Optional[List], 
        poison_data,
    ):

        # Step 1. Encoding
        embeddings, y_true = self.encode(poison_data)

        # Step 2. Clustering
        y_pred = self.clustering(embeddings)

        # Step 3. Filtering
        return self.filtering_detect(poison_data, y_true, y_pred)


    def encode(self, dataset):


        if not (self.load and self.model_load_path and os.path.exists(self.model_load_path)):
            self.logger.info("Training encoder for CUBE defense")
            self.encoder = self.trainer.train(self.encoder, {"train":dataset})
        
        self.logger.info("Reducing the dimension of hidden states")
        dataloader = get_dataloader(dataset, shuffle=False)
        hidden_states, labels, _ = self.trainer.compute_hidden(self.encoder, dataloader)
        embeddings = self.trainer.dimension_reduction(hidden_states, min_dist=0)

        return embeddings, labels


    def clustering(
        self, 
        embeddings,
        cluster_selection_epsilon: Optional[float] = 0,
        min_samples: Optional[int] = 100):

        self.logger.info("Clustering the low dimensional embeddings")
        dbscan = HDBSCAN(cluster_selection_epsilon=cluster_selection_epsilon, 
                        min_samples=min_samples)
        y_pred = dbscan.fit_predict(embeddings)

        return y_pred


    def filtering(self, dataset: List, y_true: List, y_pred: List):
        
        self.logger.info("Filtering suspicious samples")

        dropped_indices = []
        if isinstance(y_true[0], torch.Tensor):
            y_true = [y.item() for y in y_true]

        for true_label in set(y_true):
            
            groundtruth_samples = np.where(y_true==true_label*np.ones_like(y_true))[0]
            
            drop_scale = 0.5*len(groundtruth_samples)

            # Check the predictions for samples of this groundtruth label
            predictions = set()
            for i, pred in enumerate(y_pred):
                if i in groundtruth_samples:
                    predictions.add(pred)

            if len(predictions) > 1:
                count = pd.DataFrame(columns=['predictions'])

                for pred_label in predictions:
                    count.loc[pred_label,'predictions'] = \
                        np.sum(np.where((y_true==true_label*np.ones_like(y_true))*\
                                        (y_pred==pred_label*np.ones_like(y_pred)), 
                                    np.ones_like(y_pred), np.zeros_like(y_pred)))
                cluster_order = count.sort_values(by='predictions', ascending=True)
                
                # we always preserve the largest prediction cluster
                for pred_label in cluster_order.index.values[:-1]: 
                    item = cluster_order.loc[pred_label, 'predictions']
                    if item < drop_scale:

                        idx = np.where((y_true==true_label*np.ones_like(y_true))*\
                                        (y_pred==pred_label*np.ones_like(y_pred)))[0].tolist()

                        dropped_indices.extend(idx)

        filtered_dataset = []
        for i, data in enumerate(dataset):
            if i not in dropped_indices:
                filtered_dataset.append(data)
        
        return filtered_dataset

    def filtering_detect(self, dataset: List, y_true: List, y_pred: List):
        
        self.logger.info("Filtering suspicious samples")

        dropped_indices = []
        if isinstance(y_true[0], torch.Tensor):
            y_true = [y.item() for y in y_true]

        for true_label in set(y_true):
            
            groundtruth_samples = np.where(y_true==true_label*np.ones_like(y_true))[0]
            
            drop_scale = 0.5*len(groundtruth_samples)

            # Check the predictions for samples of this groundtruth label
            predictions = set()
            for i, pred in enumerate(y_pred):
                if i in groundtruth_samples:
                    predictions.add(pred)

            if len(predictions) > 1:
                count = pd.DataFrame(columns=['predictions'])

                for pred_label in predictions:
                    count.loc[pred_label,'predictions'] = \
                        np.sum(np.where((y_true==true_label*np.ones_like(y_true))*\
                                        (y_pred==pred_label*np.ones_like(y_pred)), 
                                    np.ones_like(y_pred), np.zeros_like(y_pred)))
                cluster_order = count.sort_values(by='predictions', ascending=True)
                
                # we always preserve the largest prediction cluster
                for pred_label in cluster_order.index.values[:-1]: 
                    item = cluster_order.loc[pred_label, 'predictions']
                    if item < drop_scale:

                        idx = np.where((y_true==true_label*np.ones_like(y_true))*\
                                        (y_pred==pred_label*np.ones_like(y_pred)))[0].tolist()

                        dropped_indices.extend(idx)

        # filtered_dataset = []
        # for i, data in enumerate(dataset):
        #     if i not in dropped_indices:
        #         filtered_dataset.append(data)
        
        flags = np.zeros(len(dataset))
        flags[dropped_indices] = 1
        
        return flags