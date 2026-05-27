import torch
import torch.nn as nn
from .victim import Victim
from typing import *
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from collections import namedtuple
from torch.nn.utils.rnn import pad_sequence
import os


class PLMVictim(Victim):
    """
    PLM victims. Support Huggingface's Transformers.

    Args:
        device (:obj:`str`, optional): The device to run the model on. Defaults to "gpu".
        model (:obj:`str`, optional): The model to use. Defaults to "bert".
        path (:obj:`str`, optional): The path to the model. Defaults to "bert-base-uncased".
        num_classes (:obj:`int`, optional): The number of classes. Defaults to 2.
        max_len (:obj:`int`, optional): The maximum length of the input. Defaults to 512.
    """
    def __init__(
        self, 
        device: Optional[str] = "gpu",
        model_name: Optional[str] = "bert-base-uncased",
        # model_name: Optional[str] = "bert-base-uncased",
        load_path: Optional[str] = None,
        load: Optional[bool] = False,
        num_classes: Optional[int] = 2,
        max_len: Optional[int] = 512,
        **kwargs
    ):
        super().__init__()

        self.model_name = model_name

        self.device = torch.device("cuda" if torch.cuda.is_available() and device == "gpu" else "cpu")
        # you can change huggingface model_config here
        if load and load_path and os.path.exists(load_path):
            self.model_config = AutoConfig.from_pretrained(load_path)
            self.plm = AutoModelForSequenceClassification.from_pretrained(load_path, config=self.model_config)
        else:
            self.model_config = AutoConfig.from_pretrained(model_name)
            self.model_config.num_labels = num_classes
            self.plm = AutoModelForSequenceClassification.from_pretrained(model_name, config=self.model_config)

        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.to(self.device)
        
    def to(self, device):
        self.plm = self.plm.to(device)

    def forward(self, inputs):
        output = self.plm(**inputs, output_hidden_states=True)
        return output

    def get_repr_embeddings(self, inputs):

        if self.model_name == "bert-base-uncased":
            output = getattr(self.plm, self.plm.base_model_prefix)(**inputs).last_hidden_state # batch_size, max_len, 768(1024)
            return output[:, 0, :]
        elif self.model_name == "roberta-base":

            model_output = getattr(self.plm, self.plm.base_model_prefix)(**inputs).last_hidden_state
            
            # 2. Get the attention mask to identify non-padding tokens
            # mask shape: [batch_size, seq_len]
            attention_mask = inputs['attention_mask']
            
            # 3. Expand the mask to match the hidden state dimensions
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(model_output.size()).float()
            
            # 4. Sum the embeddings and divide by the number of non-padding tokens
            sum_embeddings = torch.sum(model_output * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            
            return sum_embeddings / sum_mask
        else:
            raise NotImplementedError(f"Model {self.model_name} not supported for representation extraction.")


    def process(self, batch):
        text = batch["text"]
        labels = batch["label"]
        input_batch = self.tokenizer(text, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt").to(self.device)
        labels = labels.to(self.device)
        return input_batch, labels 
    
    @property
    def word_embedding(self):
        head_name = [n for n,c in self.plm.named_children()][0]
        layer = getattr(self.plm, head_name)
        return layer.embeddings.word_embeddings.weight
    
