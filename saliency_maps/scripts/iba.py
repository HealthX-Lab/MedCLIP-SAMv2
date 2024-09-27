"""
Based on code of https://github.com/bazingagin/IBA, https://github.com/BioroboticsLab/IBA
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from scripts.utils import replace_layer, normalize, mySequential
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Estimator:
    """
    Useful to calculate the empirical mean and variance of intermediate feature maps.
    """
    def __init__(self, layer):
        self.layer = layer
        self.M = None  # running mean for each entry
        self.S = None  # running std for each entry
        self.N = None  # running num_seen for each entry
        self.num_seen = 0  # total samples seen
        self.eps = 1e-5

    def feed(self, z: np.ndarray):

        # Initialize if this is the first datapoint
        if self.N is None:
            self.M = np.zeros_like(z, dtype=float)
            self.S = np.zeros_like(z, dtype=float)
            self.N = np.zeros_like(z, dtype=float)

        self.num_seen += 1

        diff = (z - self.M)
        self.N += 1
        self.M += diff / self.num_seen
        self.S += diff * (z - self.M)

    def feed_batch(self, batch: np.ndarray):
        for point in batch:
            self.feed(point)

    def shape(self):
        return self.M.shape

    def is_complete(self):
        return self.num_seen > 0

    def get_layer(self):
        return self.layer

    def mean(self):
        return self.M.squeeze()

    def p_zero(self):
        return 1 - self.N / (self.num_seen + 1)  # Adding 1 for stablility, so that p_zero > 0 everywhere

    def std(self, stabilize=True):
        if stabilize:
            # Add small numbers, so that dead neurons are not a problem
            return np.sqrt(np.maximum(self.S, self.eps) / np.maximum(self.N, 1.0))

        else:
            return np.sqrt(self.S / self.N)

    def estimate_density(self, z):
        z_norm = (z - self.mean()) / self.std()
        p = z_norm.pdf(z_norm, 0, 1)
        return p

    def normalize(self, z):
        return (z - self.mean()) / self.std()

    def load(self, what):
        state = what if not isinstance(what, str) else torch.load(what)
        # Check if estimator classes match
        if self.__class__.__name__ != state["class"]:
            raise RuntimeError("This Estimator is {}, cannot load {}".format(self.__class__.__name__, state["class"]))
        # Check if layer classes match
        if self.layer.__class__.__name__ != state["layer_class"]:
            raise RuntimeError("This Layer is {}, cannot load {}".format(self.layer.__class__.__name__, state["layer_class"]))
        self.N = state["N"]
        self.S = state["S"]
        self.M = state["M"]
        self.num_seen = state["num_seen"]


class InformationBottleneck(nn.Module):
    def __init__(self, mean: np.ndarray, std: np.ndarray, device=None):
        super().__init__()
        self.device = device
        self.initial_value = 5.0
        self.std = torch.tensor(std, dtype=torch.float, device=self.device, requires_grad=False)
        self.mean = torch.tensor(mean, dtype=torch.float, device=self.device, requires_grad=False)
        self.alpha = nn.Parameter(torch.full((1, *self.mean.shape), fill_value=self.initial_value, device=self.device))
        self.sigmoid = nn.Sigmoid()
        self.buffer_capacity = None

        self.reset_alpha()

    @staticmethod
    def _sample_t(mu, noise_var):
        #log_noise_var = torch.clamp(log_noise_var, -10, 10)
        noise_std = noise_var.sqrt()
        eps = mu.data.new(mu.size()).normal_()
        return mu + noise_std * eps

    @staticmethod
    def _calc_capacity(mu, var):
        # KL[P(t|x)||Q(t)] where Q(t) is N(0,1)
        kl =  -0.5 * (1 + torch.log(var) - mu**2 - var)
        return kl

    def reset_alpha(self):
        with torch.no_grad():
            self.alpha.fill_(self.initial_value)
        return self.alpha

    def forward(self, x, **kwargs):
        lamb = self.sigmoid(self.alpha)
        lamb = lamb.expand(x.shape[0], x.shape[1], -1)
        masked_mu = x * lamb
        masked_var = (1-lamb)**2
        self.buffer_capacity = self._calc_capacity(masked_mu, masked_var)
        t = self._sample_t(masked_mu, masked_var)
        return (t,)


class IBAInterpreter:
    def __init__(self, model, estim: Estimator, beta, steps=10, lr=1, batch_size=10, ensemble=False, progbar=False):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.original_layer = estim.get_layer()
        self.shape = estim.shape()
        self.beta = beta
        self.batch_size = batch_size
        self.fitting_estimator = torch.nn.CosineSimilarity(eps=1e-6)
        self.progbar = progbar
        self.lr = lr
        self.train_steps = steps
        self.bottleneck = InformationBottleneck(estim.mean(), estim.std(), device=self.device)
        self.sequential = mySequential(self.original_layer, self.bottleneck)
        self.ensemble = ensemble

    def text_heatmap(self, text_t, image_t):
        saliency, loss_c, loss_f, loss_t = self._run_text_training(text_t, image_t)
        saliency = torch.nansum(saliency, -1).cpu().detach().numpy()
        saliency = normalize(saliency)
        return normalize(saliency)
    
    def vision_heatmap(self, text_t, image_t):
        saliency, loss_c, loss_f, loss_t = self._run_vision_training(text_t, image_t)
        saliency = torch.nansum(saliency, -1)[1:] # Discard the first because it's the CLS token
        dim = int(saliency.numel() ** 0.5)
        saliency = saliency.reshape(1, 1, dim, dim)
        saliency = torch.nn.functional.interpolate(saliency, size=224, mode='bilinear')
        saliency = saliency.squeeze().cpu().detach().numpy()
        return normalize(saliency)

    def _run_text_training(self, text_t, image_t):
        replace_layer(self.model.text_model, self.original_layer, self.sequential)
        loss_c, loss_f, loss_t = self._train_bottleneck(text_t, image_t)
        replace_layer(self.model.text_model, self.sequential, self.original_layer)
        return self.bottleneck.buffer_capacity.mean(axis=0), loss_c, loss_f, loss_t
    
    def _run_vision_training(self, text_t, image_t):
        replace_layer(self.model.vision_model, self.original_layer, self.sequential)
        loss_c, loss_f, loss_t = self._train_bottleneck(text_t, image_t)
        replace_layer(self.model.vision_model, self.sequential, self.original_layer)
        return self.bottleneck.buffer_capacity.mean(axis=0), loss_c, loss_f, loss_t

    def _train_bottleneck(self, text_t: torch.Tensor, image_t: torch.Tensor):

        if(not self.ensemble):
            batch = text_t.expand(self.batch_size, -1), image_t.expand(self.batch_size, -1, -1, -1)
            optimizer = torch.optim.Adam(lr=self.lr, params=self.bottleneck.parameters())
            # Reset from previous run or modifications
            self.bottleneck.reset_alpha()
            # Train
            self.model.eval()
            for _ in tqdm(range(self.train_steps), desc="Training Bottleneck",
                        disable=not self.progbar):
                optimizer.zero_grad()
                out = self.model.get_text_features(batch[0]), self.model.get_image_features(batch[1])
                loss_c, loss_f, loss_t = self.calc_loss(outputs=out[0], labels=out[1])
                loss_t.backward()
                optimizer.step(closure=None)

        else:

            optimizer = torch.optim.Adam(lr=self.lr, params=self.bottleneck.parameters())
            image_batch = image_t.expand(self.batch_size, -1, -1, -1)
            # Reset from previous run or modifications
            self.bottleneck.reset_alpha()
            # Train
            self.model.eval()
            for _ in tqdm(range(self.train_steps), desc="Training Bottleneck",
                            disable=not self.progbar):
                optimizer.zero_grad()
                image_output = self.model.get_image_features(image_t)
                text_outputs = self.model.get_text_features(text_t)
                text_outputs = text_outputs.mean(dim=0).unsqueeze(0)
                text_outputs = text_outputs / text_outputs.norm(dim=-1, keepdim=True)
                # print(text_outputs.shape)
                # print(text_outputs.expand(self.batch_size,-1).shape)
                loss_c, loss_f, loss_t = self.calc_loss(outputs=text_outputs.expand(self.batch_size,-1), labels=image_output)
                loss_t.backward()
                optimizer.step(closure=None)

        return loss_c, loss_f, loss_t 

    def calc_loss(self, outputs, labels):
        """ Calculate the combined loss expression for optimization of lambda """
        compression_term = self.bottleneck.buffer_capacity.mean()
        fitting_term = self.fitting_estimator(outputs, labels).mean()
        total =  self.beta * compression_term - fitting_term
        return compression_term, fitting_term, total

