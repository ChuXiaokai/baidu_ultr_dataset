"""Training and testing the inverse propensity weighting algorithm for unbiased learning to rank.

See the following paper for more information on the inverse propensity weighting algorithm.

    * Xuanhui Wang, Michael Bendersky, Donald Metzler, Marc Najork. 2016. Learning to Rank with Selection Bias in Personal Search. In Proceedings of SIGIR '16
    * Thorsten Joachims, Adith Swaminathan, Tobias Schnahel. 2017. Unbiased Learning-to-Rank with Biased Feedback. In Proceedings of WSDM '17

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F
import torch

from baseline_model.learning_algorithm.base_algorithm import BaseAlgorithm
import baseline_model.utils as utils
import baseline_model as ultra


def selu(x):
    # with tf.name_scope('selu') as scope:
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * torch.where(x >= 0.0, x, alpha * F.elu(x))


class IPWrank(BaseAlgorithm):
    """The Inverse Propensity Weighting algorithm for unbiased learning to rank.

    This class implements the training and testing of the Inverse Propensity Weighting algorithm for unbiased learning to rank. See the following paper for more information on the algorithm.

    * Xuanhui Wang, Michael Bendersky, Donald Metzler, Marc Najork. 2016. Learning to Rank with Selection Bias in Personal Search. In Proceedings of SIGIR '16
    * Thorsten Joachims, Adith Swaminathan, Tobias Schnahel. 2017. Unbiased Learning-to-Rank with Biased Feedback. In Proceedings of WSDM '17

    """

    def __init__(self, exp_settings, encoder_model):
        """Create the model.

        Args:
            exp_settings: (dictionary) The dictionary containing the model settings.
        """

        self.hparams = utils.hparams.HParams(
            propensity_estimator_type='baseline_model.utils.propensity_estimator.RandomizedPropensityEstimator',
            propensity_estimator_json='baseline_model/randomized_pbm_0.1_1.0_4_1.0.json',
            learning_rate=exp_settings['lr'],                 # Learning rate.
            max_gradient_norm=0.5,            # Clip gradients to this norm.
            loss_func='sigmoid_loss',   
            # Set strength for L2 regularization.
            l2_loss=0.0,
            grad_strategy='ada',            # Select gradient strategy
        )

        self.hparams.parse(exp_settings['learning_algorithm_hparams'])
        self.exp_settings = exp_settings
        if 'selection_bias_cutoff' in self.exp_settings.keys():
            self.rank_list_size = self.exp_settings['selection_bias_cutoff']
        self.letor_features_name = "letor_features"
        self.letor_features = None
        self.feature_size = exp_settings["feature_size"]

        # DataParallel
        self.model = encoder_model
        if torch.cuda.device_count() >= exp_settings['n_gpus'] > 1:
            print("Let's use", exp_settings['n_gpus'], "GPUs!")
            self.model = nn.DataParallel(self.model, device_ids = list(range(exp_settings['n_gpus'])))
        self.model.cuda()

        if exp_settings['init_parameters'] != "":
            print('load ', exp_settings['init_parameters'])
            self.model.load_state_dict(torch.load(exp_settings['init_parameters'] ), strict=False)

        # propensity_estimator
        self.propensity_estimator = ultra.utils.find_class(
            self.hparams.propensity_estimator_type)(
            self.hparams.propensity_estimator_json
        )

        self.max_candidate_num = exp_settings['max_candidate_num']
        self.learning_rate = float(self.hparams.learning_rate)
        self.global_step = 0

        # Feeds for inputs.
        self.docid_inputs_name = []  # a list of top documents
        self.labels_name = []  # the labels for the documents (e.g., clicks)
        self.docid_inputs = []  # a list of top documents
        self.labels = []  # the labels for the documents (e.g., clicks)
        for i in range(self.max_candidate_num):
            self.docid_inputs_name.append("docid_input{0}".format(i))
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
        # Output feed: depends on whether we do a backward step or not.
        # compute propensity weights for the input data.
        self.global_step += 1
        pw = []
        self.model.train()
        for l in range(self.rank_list_size):
            input_feed["propensity_weights{0}".format(l)] = []
        for i in range(len(input_feed[self.labels_name[0]])):
            click_list = [input_feed[self.labels_name[l]][i]
                          for l in range(self.rank_list_size)]
            pw_list = self.propensity_estimator.getPropensityForOneList(
                click_list, use_non_clicked_data=True)
            pw.append(pw_list)
            for l in range(self.rank_list_size):
                input_feed["propensity_weights{0}".format(l)].append(
                    pw_list[l])

        self.propensity_weights = pw
        # Gradients and SGD update operation for training the model.

        # start train
        src = input_feed['src']
        src_segment = input_feed['src_segment']
        src_padding_mask = input_feed['src_padding_mask']
        train_output = self.model(src, src_segment, src_padding_mask)
        train_output = train_output.reshape(-1, self.max_candidate_num)

        # start optimize
        self.create_input_feed(input_feed, self.rank_list_size)

        train_labels = self.labels
        train_pw = torch.as_tensor(self.propensity_weights).cuda()
        self.loss = None

        if self.hparams.loss_func == 'sigmoid_loss':
            self.loss = self.sigmoid_loss_on_list(
                train_output, train_labels, train_pw)
        elif self.hparams.loss_func == 'pairwise_loss':
            self.loss = self.pairwise_loss_on_list(
                train_output, train_labels, train_pw)
        else:
            self.loss = self.softmax_loss(
                train_output, train_labels, train_pw)

        params = self.model.parameters()
        if self.hparams.l2_loss > 0:
            for p in params:
                self.loss += self.hparams.l2_loss * self.l2_loss(p)
        
        self.opt_step(self.optimizer_func, params)
        nn.utils.clip_grad_value_(train_labels, 1)

        # print(" Loss %f at Global Step %d: " % (self.loss.item(),self.global_step))
        return self.loss.item()

    def get_scores(self, input_feed):
        self.model.eval()
        src = input_feed['src']
        src_segment = input_feed['src_segment']
        src_padding_mask = input_feed['src_padding_mask']
        scores = self.model(src, src_segment, src_padding_mask)
        return scores
