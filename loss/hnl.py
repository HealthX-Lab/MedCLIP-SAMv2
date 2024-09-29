import torch.nn as nn
import torch.nn.functional as F
import torch

class HardNegativeLoss(nn.Module):
    """
    Hard Negative Noise Contrastive Estimation proposed in https://arxiv.org/abs/2301.02280
    beta1: hardness parameter for image features
    beta2: hardness parameter for text features
    alpha: the weighting function of the positive sample loss
    Setting alpha to 0, the loss is equivalent to the decoupled HN-NCE loss (DHN-NCE)
    temperature: temperature to control the sharpness of the distribution
    """
    def __init__(self, temperature=1.0,beta1=1.0, beta2 = 1.0, alpha=0):
        super(HardNegativeLoss, self).__init__()
        self.temperature = temperature
        self.beta1 = beta1
        self.beta2 = beta2
        self.alpha = alpha

    def forward(self, image_features, text_features,batch_size):
        # Normalize features
        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)

        # Compute cosine similarity between image and text features
        logits_per_image = torch.matmul(image_features, text_features.t()) / self.temperature
        logits_per_text = logits_per_image.t()

        mask = torch.eye(logits_per_image.size(0), dtype=torch.bool)
        mask = mask.to(image_features.device)

        # Positive pairs: diagonal elements
        pos = torch.exp(logits_per_image*mask)

        # Negative pairs: off-diagonal elements
        N = batch_size - 1

        neg_mask = ~mask

        # Calculate reweighting factors
        norm_term_img = torch.sum(torch.exp(logits_per_image*neg_mask),dim=-1)
        reweight_img = N * (torch.exp(self.beta1*logits_per_image*neg_mask))/norm_term_img
        norm_term_text = torch.sum(torch.exp(logits_per_text*neg_mask),dim=-1)
        reweight_text = N * (torch.exp(self.beta2*logits_per_text*neg_mask))/norm_term_text

        neg_img = reweight_img * torch.exp(logits_per_image*neg_mask)
        neg_text = reweight_text * torch.exp(logits_per_text*neg_mask)

        # Calculate loss
        loss = -torch.log(pos / (pos*self.alpha + neg_img)) -torch.log(pos / (pos*self.alpha + neg_text))

        return loss.mean()
    