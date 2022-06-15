"""Training and testing the Pairwise Debiasing algorithm for unbiased learning to rank.

See the following paper for more information on the Pairwise Debiasing algorithm.

    * Hu, Ziniu, Yang Wang, Qu Peng, and Hang Li. "Unbiased LambdaMART: An Unbiased Pairwise Learning-to-Rank Algorithm." In The World Wide Web Conference, pp. 2830-2836. ACM, 2019.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from baseline_model.learning_algorithm.base_algorithm import BaseAlgorithm
import baseline_model.utils as utils


def get_bernoulli_sample(probs):
    """Conduct Bernoulli sampling according to a specific probability distribution.

        Args:
            prob: (tf.Tensor) A tensor in which each element denotes a probability of 1 in a Bernoulli distribution.

        Returns:
            A Tensor of binary samples (0 or 1) with the same shape of probs.

        """
    return torch.ceil(probs - torch.rand(probs.shape).to(device=torch.device('cuda')))


class PairDebias(BaseAlgorithm):
    """The Pairwise Debiasing algorithm for unbiased learning to rank.

    This class implements the Pairwise Debiasing algorithm based on the input layer
    feed. See the following paper for more information on the algorithm.

    * Hu, Ziniu, Yang Wang, Qu Peng, and Hang Li. "Unbiased LambdaMART: An Unbiased Pairwise Learning-to-Rank Algorithm." In The World Wide Web Conference, pp. 2830-2836. ACM, 2019.

    """

    def __init__(self, exp_settings, encoder_model):
        """Create the model.

        Args:
            data_set: (Raw_data) The dataset used to build the input layer.
            exp_settings: (dictionary) The dictionary containing the model settings.
        """
        print('Build Pairwise Debiasing algorithm.')

        self.hparams = utils.hparams.HParams(
            EM_step_size=exp_settings['lr'],                  # Step size for EM algorithm.
            learning_rate=0.005,                 # Learning rate.
            max_gradient_norm=0.5,            # Clip gradients to this norm.
            # An int specify the regularization term.
            regulation_p=1,
            # Set strength for L2 regularization.
            l2_loss=0.0,
            grad_strategy='ada',            # Select gradient strategy
        )

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

        self.max_candidate_num = exp_settings['max_candidate_num']
        self.learning_rate =  float(self.hparams.learning_rate)

        # Feeds for inputs.
        self.docid_inputs_name = []  # a list of top documents
        self.labels_name = []  # the labels for the documents (e.g., clicks)
        self.docid_inputs = []  # a list of top documents
        self.labels = []  # the labels for the documents (e.g., clicks)
        for i in range(self.max_candidate_num):
            self.labels_name.append("label{0}".format(i))

        self.global_step = 0
        if 'selection_bias_cutoff' in self.exp_settings:
            self.rank_list_size = self.exp_settings['selection_bias_cutoff']
            self.t_plus = torch.ones([1, self.rank_list_size])
            self.t_minus = torch.ones([1, self.rank_list_size])

            self.t_plus = torch.ones([1, self.rank_list_size]).cuda()
            self.t_minus = torch.ones([1, self.rank_list_size]).cuda()
            self.t_plus.requires_grad = False
            self.t_minus.requires_grad = False

        # Select optimizer
        self.optimizer_func = torch.optim.Adagrad(self.model.parameters(), lr=self.hparams.learning_rate)
        if self.hparams.grad_strategy == 'sgd':
            self.optimizer_func = torch.optim.SGD(self.model.parameters(), lr=self.hparams.learning_rate)

    def train(self, input_feed):
        """Run a step of the model feeding the given inputs for training process.

        Args:
            input_feed: (dictionary) A dictionary containing all the input feed data.

        Returns:
            A triple consisting of the loss, outputs (None if we do backward),
            and a tf.summary containing related information about the step.

        """
        self.labels = []
        self.model.train()
        for i in range(self.rank_list_size):
            self.labels.append(input_feed[self.labels_name[i]])
        self.labels = torch.as_tensor(self.labels).cuda()

        
        src = input_feed['src']
        src_segment = input_feed['src_segment']
        src_padding_mask = input_feed['src_padding_mask']
        train_output = self.model(src, src_segment, src_padding_mask)
        train_output = train_output.reshape(-1, self.max_candidate_num)

        self.splitted_t_plus = torch.split(
            self.t_plus, 1, dim=1)
        self.splitted_t_minus = torch.split(
            self.t_minus, 1, dim=1)

        split_size = int(train_output.shape[1] / self.rank_list_size)
        output_list = torch.split(train_output, split_size, dim=1)
        t_plus_loss_list = [0.0 for _ in range(self.rank_list_size)]
        t_minus_loss_list = [0.0 for _ in range(self.rank_list_size)]
        self.loss = 0.0
        for i in range(self.rank_list_size):
            for j in range(self.rank_list_size):
                if i == j:
                    continue
                valid_pair_mask = torch.minimum(
                    torch.ones_like(
                        self.labels[i]), F.relu(self.labels[i] - self.labels[j]))
                pair_loss = torch.sum(
                    valid_pair_mask *
                    self.pairwise_cross_entropy_loss(
                        output_list[i], output_list[j])
                )
                t_plus_loss_list[i] += pair_loss / self.splitted_t_minus[j]
                t_minus_loss_list[j] += pair_loss / self.splitted_t_plus[i]
                self.loss += pair_loss / \
                             self.splitted_t_plus[i] / self.splitted_t_minus[j]

        with torch.no_grad():
            self.t_plus = (1 - self.hparams.EM_step_size) * self.t_plus + self.hparams.EM_step_size * torch.pow(
                    torch.cat(t_plus_loss_list, dim=1) / t_plus_loss_list[0], 1 / (self.hparams.regulation_p + 1))
            self.t_minus = (1 - self.hparams.EM_step_size) * self.t_minus + self.hparams.EM_step_size * torch.pow(torch.cat(
                    t_minus_loss_list, dim=1) / t_minus_loss_list[0], 1 / (self.hparams.regulation_p + 1))

        # Add l2 loss
        params = self.model.parameters()
        if self.hparams.l2_loss > 0:
            for p in params:
                self.loss += self.hparams.l2_loss * self.l2_loss(p)

        self.opt_step(self.optimizer_func, params)
        self.global_step+=1
        return self.loss.item()


    def get_scores(self, input_feed):
        self.model.eval()
        src = input_feed['src']
        src_segment = input_feed['src_segment']
        src_padding_mask = input_feed['src_padding_mask']
        scores = self.model(src, src_segment, src_padding_mask)
        return scores
