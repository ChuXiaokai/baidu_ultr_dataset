"""The navie algorithm that directly trains ranking models with clicks.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


from baseline_model.learning_algorithm.base_algorithm import BaseAlgorithm
import baseline_model.utils as utils


class NavieAlgorithm(BaseAlgorithm):
    """The navie algorithm that directly trains ranking models with input labels.

    """

    def __init__(self, exp_settings, encoder_model):
        """Create the model.

        Args:
            data_set: (Raw_data) The dataset used to build the input layer.
        """
        print('Build NavieAlgorithm')

        self.hparams = utils.hparams.HParams(
            learning_rate=exp_settings['lr'],                 # Learning rate.
            max_gradient_norm=0.5,            # Clip gradients to this norm.
            loss_func='sigmoid_loss',            # Select Loss function
            # Set strength for L2 regularization.
            l2_loss=0.0,
            grad_strategy='ada',            # Select gradient strategy
        )
        self.train_summary = {}
        self.eval_summary = {}
        self.is_training = "is_train"
        print(exp_settings['learning_algorithm_hparams'])
        self.hparams.parse(exp_settings['learning_algorithm_hparams'])
        self.exp_settings = exp_settings
        if 'selection_bias_cutoff' in self.exp_settings.keys():
            self.rank_list_size = self.exp_settings['selection_bias_cutoff']
        self.feature_size = exp_settings['feature_size']

        # DataParallel
        self.model = encoder_model
        if torch.cuda.device_count() >= exp_settings['n_gpus'] > 1:
            print("Let's use", exp_settings['n_gpus'], "GPUs!")
            self.model = nn.DataParallel(self.model, device_ids = list(range(exp_settings['n_gpus'])))
        self.model.cuda()

        if exp_settings['init_parameters'] != "":
            print('load ', exp_settings['init_parameters'])
            self.model.load_state_dict(torch.load(exp_settings['init_parameters'] ), strict=False)

        self.max_candidate_num = exp_settings['max_candidate_num']
        self.learning_rate = float(self.hparams.learning_rate)
        self.global_step = 0

        # Feeds for inputs.
        self.letor_features_name = "letor_features"
        self.letor_features = None
        self.labels_name = []  # the labels for the documents (e.g., clicks)
        self.docid_inputs = []  # a list of top documents
        self.labels = []  # the labels for the documents (e.g., clicks)
        for i in range(self.max_candidate_num):
            self.labels_name.append("label{0}".format(i))

        self.optimizer_func = torch.optim.Adagrad(self.model.parameters(), lr=self.learning_rate)
        if self.hparams.grad_strategy == 'sgd':
            self.optimizer_func = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def train(self, input_feed):
        """Run a step of the model feeding the given inputs for training process.

        Args:
            input_feed: (dictionary) A dictionary containing all the input feed data.

        Returns:
            A triple consisting of the loss, outputs (None if we do backward),
            and a tf.summary containing related information about the step.

        """
        self.global_step += 1
        self.model.train()
        self.create_input_feed(input_feed, self.rank_list_size)

        # Gradients and SGD update operation for training the model.
        src = input_feed['src']
        src_segment = input_feed['src_segment']
        src_padding_mask = input_feed['src_padding_mask']
        train_output = self.model(src, src_segment, src_padding_mask)
        train_output = train_output.reshape(-1, self.max_candidate_num)
        train_labels = self.labels
        self.loss = None

        if self.hparams.loss_func == 'sigmoid_loss':
            self.loss = self.sigmoid_loss_on_list(
                train_output, train_labels)
        elif self.hparams.loss_func == 'pairwise_loss':
            self.loss = self.pairwise_loss_on_list(
                train_output, train_labels)
        else:
            self.loss = self.softmax_loss(
                train_output, train_labels)

        params = self.model.parameters()
        if self.hparams.l2_loss > 0:
            loss_l2 = 0.0
            for p in params:
                loss_l2 += self.l2_loss(p)
            self.loss += self.hparams.l2_loss * loss_l2

        self.opt_step(self.optimizer_func, params)

        nn.utils.clip_grad_value_(train_labels, 1)
        return self.loss.item()

    def get_scores(self, input_feed):
        self.model.eval()
        src = input_feed['src']
        src_segment = input_feed['src_segment']
        src_padding_mask = input_feed['src_padding_mask']
        scores = self.model(src, src_segment, src_padding_mask)
        return scores

