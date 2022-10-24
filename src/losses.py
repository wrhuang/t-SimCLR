from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimCLRLoss(nn.Module):
    """
    SimCLR: https://arxiv.org/pdf/2002.05709.pdf
    """
    def __init__(self, temperature=0.5, base_temperature=0.07):
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features):
        """Args:
            features: hidden vector of shape [bsz, n_views, feat_dim].
        """
        if not len(features.shape) == 3:
            raise ValueError('`features` needs to be [bsz, n_views, feat_dim],'
                             '3 dimensions are required')
        
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu')) 
        
        features = F.normalize(features, dim=2)
        batch_size = features.shape[0]
        mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        
        contrast_count = features.shape[1]  # n_views
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # [bsz*n_views, feat_dim]

        # compute logits
        contrast_dot_contrast = torch.div(
            torch.matmul(contrast_feature, contrast_feature.T),
            self.temperature)   # logits, [bsz*n_views, bsz*n_views]

        # for numerical stability
        logits_max, _ = torch.max(contrast_dot_contrast, dim=1, keepdim=True)
        logits = contrast_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(contrast_count, contrast_count)  # n_views*n_views identity matrice, each shape is [bsz, bsz]
        
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * contrast_count).view(-1, 1).to(device),
            0
        )   # [bsz*n_views, bsz*n_views], value = 1-I
        mask = mask * logits_mask   # [bsz*n_views, bsz*n_views], value = [0,I; I,0]

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        """ The absolute value of loss decreases with increasing temperature.
            To make the losses get similar values with different temperature, loss is multiplied by temp/base_temp """
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos 
        loss = loss.view(contrast_count, batch_size).mean()

        return loss

class t_SimCLRLoss(nn.Module):
    """
    t-SimCLR: https://arxiv.org/pdf/2205.14814.pdf
    """
    def __init__(self, temperature=0.5, t_df=5):
        super(t_SimCLRLoss, self).__init__()
        self.temperature = temperature
        self.t_df = t_df

    def forward(self, features):
        """Args:
            features: hidden vector of shape [bsz, n_views, feat_dim].
        """
        if not len(features.shape) == 3:
            raise ValueError('`features` needs to be [bsz, n_views, feat_dim],'
                             '3 dimensions are required')
        
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))       

        batch_size = features.shape[0]
        mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        
        contrast_count = features.shape[1]  # n_views
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # [bsz*n_views, feat_dim]

        # compute logits
        # features_square = torch.square(torch.norm(features, dim=2)) # [bsz, n_views]

        contrast_feature_matrix = contrast_feature.repeat(batch_size*contrast_count,1,1)   # [bsz*n_views, bsz*n_views, feat_dim]
        contrast_feature_matrix = contrast_feature_matrix - contrast_feature_matrix.transpose(0,1)
        features_square = torch.square(torch.norm(contrast_feature_matrix, dim=2)) # [bsz*n_views, bsz*n_views]
        logits = (torch.div(features_square, self.temperature * self.t_df) + 1) 
        # for numerical stability
        logits_max, _ = torch.max(logits)
        logits = torch.div(logits, logits_max.detach())
        logits = logits - torch.diag_embed(torch.diag(logits)-1)    # The diag elements are set to 1 to avoid inf. Note that they will not be used later.
        logits = logits ** (-(self.t_df/2+0.5))

        # tile mask
        mask = mask.repeat(contrast_count, contrast_count)  # n_views*n_views identity matrice, each shape is [bsz, bsz]
        
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * contrast_count).view(-1, 1).to(device),
            0
        )   # [bsz*n_views, bsz*n_views], value = 1-I
        mask = mask * logits_mask   # [bsz*n_views, bsz*n_views], value = [0,I; I,0]

        # compute loss
        first_term = torch.log(logits) * mask
        # first_term = -(self.t_df/2+0.5) * torch.log(torch.div(features_square, self.temperature * self.t_df) + 1) * mask
        first_term = - torch.div(torch.sum(first_term), 2*batch_size)
        second_term = torch.log(torch.sum(logits * logits_mask) + 1e-6)
        loss = first_term + second_term

        return loss
