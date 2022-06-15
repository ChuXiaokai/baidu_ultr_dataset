"""The basic class that contains all the API needed for the implementation of an unbiased learning to rank algorithm.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from matplotlib.cbook import print_cycles

import torch.nn.functional as F
import torch
import numpy as np
from abc import ABC, abstractmethod

import baseline_model.utils as utils

def softmax_cross_entropy_with_logits(logits, labels):
    """Computes softmax cross entropy between logits and labels.

    Args:
        output: A tensor with shape [batch_size, list_size]. Each value is
        the ranking score of the corresponding example.
        labels: A tensor of the same shape as `output`. A value >= 1 means a
        relevant example.
    Returns:
        A single value tensor containing the loss.
    """
    loss = torch.sum(- labels * F.log_softmax(logits, -1), -1)
    return loss

class BaseAlgorithm(ABC):
    """The basic class that contains all the API needed for the
        implementation of an unbiased learning to rank algorithm.

    """
    PADDING_SCORE = -100000

    @abstractmethod
    def __init__(self, exp_settings, encoder_model):
        """Create the model.

        Args:
            data_set: (Raw_data) The dataset used to build the input layer.
            exp_settings: (dictionary) The dictionary containing the model settings.
        """
        self.is_training = None
        self.docid_inputs = None  # a list of top documents
        self.letor_features = None  # the letor features for the documents
        self.labels = None  # the labels for the documents (e.g., clicks)
        self.output = None  # the ranking scores of the inputs
        # the number of documents considered in each rank list.
        self.rank_list_size = None
        # the maximum number of candidates for each query.
        self.max_candidate_num = None
        self.optimizer_func = torch.optim.adagrad()



    @abstractmethod
    def train(self, input_feed):
        """Run a step of the model feeding the given inputs for training.

        Args:
            input_feed: (dictionary) A dictionary containing all the input feed data.

        Returns:
            A triple consisting of the loss, outputs (None if we do backward),
            and a summary containing related information about the step.

        """
        pass

    def create_input_feed(self, input_feed, list_size): 
        self.labels = []
        for i in range(list_size):
            self.labels.append(input_feed[self.labels_name[i]])
        self.labels = np.transpose(self.labels)
        self.labels = torch.FloatTensor(self.labels).cuda()


    def opt_step(self, opt, params):
        """ Perform an optimization step

        Args:
            opt: Optimization Function to use
            params: Model's parameters

        Returns
            The ranking model that will be used to computer the ranking score.

        """
        opt.zero_grad()
        self.loss.backward()
        if self.hparams.max_gradient_norm > 0:
            self.clipped_gradient = torch.nn.utils.clip_grad_norm_(
                params, self.hparams.max_gradient_norm)
        opt.step()

    def pairwise_cross_entropy_loss(
            self, pos_scores, neg_scores, propensity_weights=None):
        """Computes pairwise softmax loss without propensity weighting.

        Args:
            pos_scores: (torch.Tensor) A tensor with shape [batch_size, 1]. Each value is
            the ranking score of a positive example.
            neg_scores: (torch.Tensor) A tensor with shape [batch_size, 1]. Each value is
            the ranking score of a negative example.
            propensity_weights: (torch.Tensor) A tensor of the same shape as `output` containing the weight of each element.

        Returns:
            (torch.Tensor) A single value tensor containing the loss.
        """
        if propensity_weights is None:
            propensity_weights = torch.ones_like(pos_scores)
        label_dis = torch.cat(
            [torch.ones_like(pos_scores), torch.zeros_like(neg_scores)], dim=1)
        loss = softmax_cross_entropy_with_logits(
            logits = torch.cat([pos_scores, neg_scores], dim=1), labels = label_dis)* propensity_weights
        return loss

    def sigmoid_loss_on_list(self, output, labels,
                             propensity_weights=None):
        """Computes pointwise sigmoid loss without propensity weighting.

        Args:
            output: (torch.Tensor) A tensor with shape [batch_size, list_size]. Each value is
            the ranking score of the corresponding example.
            labels: (torch.Tensor) A tensor of the same shape as `output`. A value >= 1 means a
            relevant example.
            propensity_weights: (torch.Tensor) A tensor of the same shape as `output` containing the weight of each element.

        Returns:
            (torch.Tensor) A single value tensor containing the loss.
        """
        if propensity_weights is None:
            propensity_weights = torch.ones_like(labels)
        criterion =  torch.nn.BCEWithLogitsLoss(reduction="none")
        loss = criterion(output, labels) * propensity_weights
        return torch.mean(torch.sum(loss, dim=1))

    def pairwise_loss_on_list(self, output, labels,
                              propensity_weights=None):
        """Computes pairwise entropy loss.

        Args:
            output: (torch.Tensor) A tensor with shape [batch_size, list_size]. Each value is
            the ranking score of the corresponding example.
            labels: (torch.Tensor) A tensor of the same shape as `output`. A value >= 1 means a
                relevant example.
            propensity_weights: (torch.Tensor) A tensor of the same shape as `output` containing the weight of each element.

        Returns:
            (torch.Tensor) A single value tensor containing the loss.
        """
        if propensity_weights is None:
            propensity_weights = torch.ones_like(labels)

        loss = None
        sliced_output = torch.unbind(output, dim=1)
        sliced_label = torch.unbind(labels, dim=1)
        sliced_propensity = torch.unbind(propensity_weights, dim=1)
        for i in range(len(sliced_output)):
            for j in range(i + 1, len(sliced_output)):
                cur_label_weight = torch.sign(
                    sliced_label[i] - sliced_label[j])
                cur_propensity = sliced_propensity[i] * \
                    sliced_label[i] + \
                    sliced_propensity[j] * sliced_label[j]
                cur_pair_loss = - \
                    torch.exp(
                        sliced_output[i]) / (torch.exp(sliced_output[i]) + torch.exp(sliced_output[j]))
                if loss is None:
                    loss = cur_label_weight * cur_pair_loss
                loss += cur_label_weight * cur_pair_loss * cur_propensity
        batch_size = labels.size()[0]
        return torch.sum(loss) / batch_size.type(torch.float32)

    def softmax_loss(self, output, labels, propensity_weights=None):
        """Computes listwise softmax loss without propensity weighting.

        Args:
            output: (torch.Tensor) A tensor with shape [batch_size, list_size]. Each value is
            the ranking score of the corresponding example.
            labels: (torch.Tensor) A tensor of the same shape as `output`. A value >= 1 means a
            relevant example.
            propensity_weights: (torch.Tensor) A tensor of the same shape as `output` containing the weight of each element.

        Returns:
            (torch.Tensor) A single value tensor containing the loss.
        """
        if propensity_weights is None:
            propensity_weights = torch.ones_like(labels)

        weighted_labels = (labels + 0.0000001) * propensity_weights
        label_dis = weighted_labels / \
            torch.sum(weighted_labels, 1, keepdim=True)
        label_dis = torch.nan_to_num(label_dis)
        loss = softmax_cross_entropy_with_logits(
            logits = output, labels = label_dis)* torch.sum(weighted_labels, 1)
        return torch.sum(loss) / torch.sum(weighted_labels)

    def l2_loss(self, input):
        return torch.sum(input ** 2)/2



