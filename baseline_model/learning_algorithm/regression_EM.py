"""Training and testing the regression-based EM algorithm for unbiased learning to rank.

See the following paper for more information on the regression-based EM algorithm.

    * Wang, Xuanhui, Nadav Golbandi, Michael Bendersky, Donald Metzler, and Marc Najork. "Position bias estimation for unbiased learning to rank in personal search." In Proceedings of the Eleventh ACM International Conference on Web Search and Data Mining, pp. 610-618. ACM, 2018.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch


from baseline_model.learning_algorithm.base_algorithm import BaseAlgorithm
import baseline_model.utils as utils

def get_bernoulli_sample(probs):
    """Conduct Bernoulli sampling according to a specific probability distribution.

        Args:
            prob: (torch.Tensor) A tensor in which each element denotes a probability of 1 in a Bernoulli distribution.

        Returns:
            A Tensor of binary samples (0 or 1) with the same shape of probs.

        """
    if torch.cuda.is_available():
        bernoulli_sample = torch.ceil(probs - torch.rand(probs.shape, device=torch.device('cuda')))
    else:
        bernoulli_sample = torch.ceil(probs - torch.rand(probs.shape))
    return bernoulli_sample


class RegressionEM(BaseAlgorithm):
    """The regression-based EM algorithm for unbiased learning to rank.

    This class implements the regression-based EM algorithm based on the input layer
    feed. See the following paper for more information.

    * Wang, Xuanhui, Nadav Golbandi, Michael Bendersky, Donald Metzler, and Marc Najork. "Position bias estimation for unbiased learning to rank in personal search." In Proceedings of the Eleventh ACM International Conference on Web Search and Data Mining, pp. 610-618. ACM, 2018.

    In particular, we use the online EM algorithm for the parameter estimations:

    * Cappé, Olivier, and Eric Moulines. "Online expectation–maximization algorithm for latent data models." Journal of the Royal Statistical Society: Series B (Statistical Methodology) 71.3 (2009): 593-613.

    """

    def __init__(self, exp_settings, encoder_model):
        """Create the model.

        Args:
            exp_settings: (dictionary) The dictionary containing the model settings.
        """
        print('Build Regression-based EM algorithm.')

        self.hparams = utils.hparams.HParams(
            EM_step_size=exp_settings['lr']*10.,                  # Step size for EM algorithm.
            learning_rate=exp_settings['lr'],                 # Learning rate.
            max_gradient_norm=0.5,            # Clip gradients to this norm.
            # Set strength for L2 regularization.
            l2_loss=0.0,
            grad_strategy='ada',            # Select gradient strategy
        )
        self.hparams.parse(exp_settings['learning_algorithm_hparams'])
        self.exp_settings = exp_settings
        if 'selection_bias_cutoff' in self.exp_settings.keys():
            self.rank_list_size = self.exp_settings['selection_bias_cutoff']
        self.max_candidate_num = exp_settings['max_candidate_num']
        self.feature_size = exp_settings['feature_size']

        # DataParallel
        self.model = encoder_model
        if torch.cuda.device_count() >= exp_settings['n_gpus'] > 1:
            print("Let's use", exp_settings['n_gpus'], "GPUs!")
            self.model = nn.DataParallel(self.model, device_ids = list(range(exp_settings['n_gpus'])))
        self.model.cuda()

        self.letor_features_name = "letor_features"
        self.letor_features = None
        self.docid_inputs_name = []  # a list of top documents
        self.labels_name = []  # the labels for the documents (e.g., clicks)
        self.docid_inputs = []  # a list of top documents
        self.labels = []  # the labels for the documents (e.g., clicks)
        for i in range(self.max_candidate_num):
            self.docid_inputs_name.append("docid_input{0}".format(i))
            self.labels_name.append("label{0}".format(i))
        with torch.no_grad():
            self.propensity = (torch.ones([1, self.rank_list_size]) * 0.9)
            self.propensity = self.propensity.cuda()
        self.learning_rate = float(self.hparams.learning_rate)
        self.global_step = 0
        self.sigmoid_prob_b = (torch.ones([1]) - 1.0)
        self.sigmoid_prob_b = self.sigmoid_prob_b.cuda()
            # self.sigmoid_prob_b = self.sigmoid_prob_b.to(device=self.cuda)
        # Select optimizer
        self.optimizer_func = torch.optim.Adagrad(self.model.parameters(), lr=self.hparams.learning_rate)
        # tf.train.AdagradOptimizer
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
        self.model.train()
        self.create_input_feed(input_feed, self.rank_list_size)

        src = input_feed['src']
        src_segment = input_feed['src_segment']
        src_padding_mask = input_feed['src_padding_mask']
        train_output = self.model(src, src_segment, src_padding_mask)
        train_output = train_output.reshape(-1, self.max_candidate_num)
        train_output = train_output + self.sigmoid_prob_b

        # Conduct estimation step.
        gamma = torch.sigmoid(train_output)
        # reshape from [rank_list_size, ?] to [?, rank_list_size]
        reshaped_train_labels = self.labels
        p_e1_r0_c0 = self.propensity * \
                     (1 - gamma) / (1 - self.propensity * gamma)
        p_e0_r1_c0 = (1 - self.propensity) * gamma / \
                     (1 - self.propensity * gamma)
        p_r1 = reshaped_train_labels + \
               (1 - reshaped_train_labels) * p_e0_r1_c0

        # Get Bernoulli samples and compute rank loss
        self.ranker_labels = get_bernoulli_sample(p_r1).cuda()

        criterion = torch.nn.BCEWithLogitsLoss()

        self.loss = criterion(train_output,self.ranker_labels)
       
        params = self.model.parameters()
        if self.hparams.l2_loss > 0:
            for p in params:
                self.loss += self.hparams.l2_loss * self.l2_loss(p)

        opt = self.optimizer_func
        opt.zero_grad(set_to_none=True)
        self.loss.backward()
        if self.loss == 0:
            for name, param in self.model.named_parameters():
                print(name, param)
        if self.hparams.max_gradient_norm > 0:
            self.clipped_gradient = nn.utils.clip_grad_norm_(
                params, self.hparams.max_gradient_norm)
        opt.step()
        nn.utils.clip_grad_value_(reshaped_train_labels, 1)

        # Conduct maximization step
        with torch.no_grad():
            self.propensity = (1 - self.hparams.EM_step_size) * self.propensity + self.hparams.EM_step_size * torch.mean(
        reshaped_train_labels + (1 - reshaped_train_labels) * p_e1_r0_c0, dim=0, keepdim=True)
        self.update_propensity_op = self.propensity
        self.propensity_weights = 1.0 / self.propensity


        self.global_step += 1
        return self.loss.item()


    def get_scores(self, input_feed):
        self.model.eval()
        src = input_feed['src']
        src_segment = input_feed['src_segment']
        src_padding_mask = input_feed['src_padding_mask']
        scores = self.model(src, src_segment, src_padding_mask)
        return scores
