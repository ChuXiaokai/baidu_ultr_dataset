"""Training and testing the dual learning algorithm for unbiased learning to rank.

See the following paper for more information on the dual learning algorithm.

    * Qingyao Ai, Keping Bi, Cheng Luo, Jiafeng Guo, W. Bruce Croft. 2018. Unbiased Learning to Rank with Unbiased Propensity Estimation. In Proceedings of SIGIR '18

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch
import numpy as np

from baseline_model.learning_algorithm.base_algorithm import BaseAlgorithm
import baseline_model.utils as utils

def sigmoid_prob(logits):
    return torch.sigmoid(logits - torch.mean(logits, -1, keepdim=True))

class DenoisingNet(nn.Module):
    def __init__(self, input_vec_size):
        super(DenoisingNet, self).__init__()
        self.linear_layer = nn.Linear(input_vec_size, 1)
        self.elu_layer = nn.ELU()
        self.propensity_net = nn.Sequential(self.linear_layer, self.elu_layer)
        self.list_size = input_vec_size

    def forward(self, input_list):
        output_propensity_list = []
        for i in range(self.list_size):
            # Add position information (one-hot vector)
            click_feature = [
                torch.unsqueeze(
                    torch.zeros_like(
                        input_list[i]), -1) for _ in range(self.list_size)]
            click_feature[i] = torch.unsqueeze(
                torch.ones_like(input_list[i]), -1)
            # Predict propensity with a simple network
            output_propensity_list.append(
                self.propensity_net(
                    torch.cat(
                        click_feature, 1)))

        return torch.cat(output_propensity_list, 1)



class DLA(BaseAlgorithm):
    """The Dual Learning Algorithm for unbiased learning to rank.

    This class implements the Dual Learning Algorithm (DLA) based on the input layer
    feed. See the following paper for more information on the algorithm.

    * Qingyao Ai, Keping Bi, Cheng Luo, Jiafeng Guo, W. Bruce Croft. 2018. Unbiased Learning to Rank with Unbiased Propensity Estimation. In Proceedings of SIGIR '18

    """

    def __init__(self, exp_settings, encoder_model):
        """Create the model.

        Args:
            data_set: (Raw_data) The dataset used to build the input layer.
            exp_settings: (dictionary) The dictionary containing the model settings.
        """
        print('Build DLA')

        self.hparams = utils.hparams.HParams(
            learning_rate=exp_settings['lr'],                 # Learning rate.
            max_gradient_norm=0.5,            # Clip gradients to this norm.
            loss_func='sigmoid_loss',            # Select Loss function
            # the function used to convert logits to probability distributions
            logits_to_prob='sigmoid_loss',
            # The learning rate for ranker (-1 means same with learning_rate).
            propensity_learning_rate=-1.0,
            ranker_loss_weight=1.0,            # Set the weight of unbiased ranking loss
            # Set strength for L2 regularization.
            l2_loss=0.0,
            max_propensity_weight=-1,      # Set maximum value for propensity weights
            constant_propensity_initialization=False,
            # Set true to initialize propensity with constants.
            grad_strategy='ada',            # Select gradient strategy
        )
        self.train_summary = {}
        self.eval_summary = {}
        self.hparams.parse(exp_settings['learning_algorithm_hparams'])
        self.exp_settings = exp_settings
        self.max_candidate_num = exp_settings['max_candidate_num']
        self.feature_size = exp_settings['feature_size']


        if 'selection_bias_cutoff' in exp_settings.keys():
            self.rank_list_size = self.exp_settings['selection_bias_cutoff']
            self.propensity_model = DenoisingNet(self.rank_list_size)

        # DataParallel
        self.model = encoder_model
        if torch.cuda.device_count() >= exp_settings['n_gpus'] > 1:
            print("Let's use", exp_settings['n_gpus'], "GPUs!")
            self.model = nn.DataParallel(self.model, device_ids = list(range(exp_settings['n_gpus'])))
        self.model.cuda()

        if exp_settings['init_parameters'] != "":
            print('load ', exp_settings['init_parameters'])
            self.model.load_state_dict(torch.load(exp_settings['init_parameters'] ), strict=False)

        
        self.propensity_model = self.propensity_model.cuda()
        self.letor_features_name = "letor_features"
        self.letor_features = None
        self.labels_name = []  # the labels for the documents (e.g., clicks)
        self.docid_inputs = []  # a list of top documents
        self.labels = []  # the labels for the documents (e.g., clicks)
        for i in range(self.max_candidate_num):
            self.labels_name.append("label{0}".format(i))

        if self.hparams.propensity_learning_rate < 0:
            self.propensity_learning_rate = float(self.hparams.learning_rate)
        else:
            self.propensity_learning_rate = float(self.hparams.propensity_learning_rate)
        self.learning_rate = float(self.hparams.learning_rate)

        self.global_step = 0

        # Select logits to prob function
        self.logits_to_prob = nn.Softmax(dim=-1)
        if self.hparams.logits_to_prob == 'sigmoid':
            self.logits_to_prob = sigmoid_prob

        self.optimizer_func = torch.optim.Adagrad
        if self.hparams.grad_strategy == 'sgd':
            self.optimizer_func = torch.optim.SGD

        print('Loss Function is ' + self.hparams.loss_func)
        # Select loss function
        self.loss_func = None
        if self.hparams.loss_func == 'sigmoid_loss':
            self.loss_func = self.sigmoid_loss_on_list
        elif self.hparams.loss_func == 'pairwise_loss':
            self.loss_func = self.pairwise_loss_on_list
        else:  # softmax loss without weighting
            self.loss_func = self.softmax_loss

    def separate_gradient_update(self):
        denoise_params = self.propensity_model.parameters()
        ranking_model_params = self.model.parameters()
        # Select optimizer

        if self.hparams.l2_loss > 0:
            for p in ranking_model_params:
                self.rank_loss += self.hparams.l2_loss * self.l2_loss(p)
        self.loss = self.exam_loss + self.hparams.ranker_loss_weight * self.rank_loss

        opt_denoise = self.optimizer_func(self.propensity_model.parameters(), self.propensity_learning_rate)
        opt_ranker = self.optimizer_func(self.model.parameters(), self.learning_rate)

        opt_denoise.zero_grad()
        opt_ranker.zero_grad()

        self.loss.backward()

        if self.hparams.max_gradient_norm > 0:
            nn.utils.clip_grad_norm_(self.propensity_model.parameters(), self.hparams.max_gradient_norm)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.hparams.max_gradient_norm)

        opt_denoise.step()
        opt_ranker.step()

    def train(self, input_feed):
        """Run a step of the model feeding the given inputs.

        Args:
            input_feed: (dictionary) A dictionary containing all the input feed data.

        Returns:
            A triple consisting of the loss, outputs (None if we do backward),
            and a tf.summary containing related information about the step.

        """

        # Build model
        self.rank_list_size = self.exp_settings['selection_bias_cutoff']
        self.model.train()
        self.create_input_feed(input_feed, self.rank_list_size)

        # start train
        src = input_feed['src']
        src_segment = input_feed['src_segment']
        src_padding_mask = input_feed['src_padding_mask']
        train_output = self.model(src, src_segment, src_padding_mask)
        train_output = train_output.reshape(-1, self.max_candidate_num)
            
        self.propensity_model.train()
        propensity_labels = torch.transpose(self.labels,0,1)
        self.propensity = self.propensity_model(
            propensity_labels)
        with torch.no_grad():
            self.propensity_weights = self.get_normalized_weights(
                self.logits_to_prob(self.propensity))
        self.rank_loss = self.loss_func(
            train_output, self.labels, self.propensity_weights)

        # Compute examination loss
        with torch.no_grad():
            self.relevance_weights = self.get_normalized_weights(
                self.logits_to_prob(train_output))

        self.exam_loss = self.loss_func(
            self.propensity,
            self.labels,
            self.relevance_weights
        )
      
        self.loss = self.exam_loss + self.hparams.ranker_loss_weight * self.rank_loss
        self.separate_gradient_update()

        self.clip_grad_value(self.labels, clip_value_min=0, clip_value_max=1)
        self.global_step+=1
        return self.loss.item()

    def get_scores(self, input_feed):
        self.model.eval()
        src = input_feed['src']
        src_segment = input_feed['src_segment']
        src_padding_mask = input_feed['src_padding_mask']
        scores = self.model(src, src_segment, src_padding_mask)
        return scores


    def get_normalized_weights(self, propensity):
        """Computes listwise softmax loss with propensity weighting.

        Args:
            propensity: (tf.Tensor) A tensor of the same shape as `output` containing the weight of each element.

        Returns:
            (tf.Tensor) A tensor containing the propensity weights.
        """
        propensity_list = torch.unbind(
            propensity, dim=1)  # Compute propensity weights
        pw_list = []
        for i in range(len(propensity_list)):
            pw_i = propensity_list[0] / propensity_list[i]
            pw_list.append(pw_i)
        propensity_weights = torch.stack(pw_list, dim=1)
        if self.hparams.max_propensity_weight > 0:
            self.clip_grad_value(propensity_weights,clip_value_min=0,
                clip_value_max=self.hparams.max_propensity_weight)
        return propensity_weights

    def clip_grad_value(self, parameters, clip_value_min, clip_value_max) -> None:
        """Clips gradient of an iterable of parameters at specified value.

        Gradients are modified in-place.

        Args:
            parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
                single Tensor that will have gradients normalized
            clip_value (float or int): maximum allowed value of the gradients.
                The gradients are clipped in the range
                :math:`\left[\text{-clip\_value}, \text{clip\_value}\right]`
        """
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        clip_value_min = float(clip_value_min)
        clip_value_max = float(clip_value_max)
        for p in filter(lambda p: p.grad is not None, parameters):
            p.grad.data.clamp_(min=clip_value_min, max=clip_value_max)