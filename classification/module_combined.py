import torch
from torch import nn
from transformers import RobertaModel, GPT2Model
from utils import eos_pooling
import torch.nn.functional as F

class RobertaCBL(nn.Module):
    def __init__(self, concept_dim, dropout, class_num):
        super().__init__()
        self.preLM = RobertaModel.from_pretrained('roberta-base')
        for p in self.preLM.parameters():
            p.requires_grad = True
        self.projection = nn.Linear(768, concept_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(concept_dim, concept_dim)
        self.clf = nn.Linear(concept_dim, class_num)
        self.dropout = nn.Dropout(dropout)
        

    def forward(self, t):
        text_features = self.preLM(input_ids=t["input_ids"], attention_mask=t["attention_mask"]).last_hidden_state[:, 0, :]
        projected = self.projection(text_features)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        feature = x + projected
        
        ### clf
        x = F.relu(feature)
        x = self.clf(x)
        x = F.softmax(x, dim=-1)
        return feature, x


class RobertaCBLResidual(nn.Module):
    def __init__(self, concept_dim, residual_dim, dropout, class_num):
        super().__init__()
        self.preLM = RobertaModel.from_pretrained('roberta-base')
        self.concept_dim = concept_dim
        self.residual_dim = residual_dim
        for p in self.preLM.parameters():
            p.requires_grad = True
            
        self.projection = nn.Linear(768, concept_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(concept_dim, concept_dim)
        
        self.projection_residual = nn.Linear(768, residual_dim)
        self.gelu_residual = nn.GELU()
        self.fc_residual = nn.Linear(residual_dim, residual_dim)
        
        
        self.clf = nn.Linear(concept_dim+residual_dim, class_num)
        self.dropout = nn.Dropout(dropout)
        

    def forward(self, t):
        text_features = self.preLM(input_ids=t["input_ids"], attention_mask=t["attention_mask"]).last_hidden_state[:, 0, :]
        projected = self.projection(text_features)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        feature = x + projected
        
        projected_residual = self.projection_residual(text_features)
        x_residual = self.gelu_residual(projected_residual)
        x_residual = self.fc_residual(x_residual)
        x_residual = self.dropout(x_residual)
        feature_residual = x_residual + projected_residual
        
        feature_combined = torch.concat((feature, feature_residual), dim=-1)
        ### clf
        x = F.relu(feature_combined)
        x = self.clf(x)
        x = F.softmax(x, dim=-1)
        return feature, feature_residual, x

    def compute_residual_contrib(self, feature_residual):
        w = self.clf.weight  # shape: (class_num, concept_dim + residual_dim)
        w_non_concept = w[:, self.concept_dim:]  # shape: (class_num, residual_dim)
        contrib = F.linear(feature_residual, w_non_concept)  # shape: (batch_size, class_num)
        return contrib