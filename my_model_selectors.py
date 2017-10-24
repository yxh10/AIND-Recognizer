import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        num_states = 0
        min_bic_score = float("inf")
        best_num_components = 0

        try:
            for current_num_states in range(self.min_n_components, self.max_n_components):
                num_states = current_num_states
                hmm_model = GaussianHMM(current_num_states, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(self.X, self.lengths)

                logL = hmm_model.score(self.X, self.lengths)
                num_data_points = len(self.sequences)
                current_bic_score = -2 * logL + current_num_states * np.log(num_data_points)

                if current_bic_score < min_bic_score:
                    min_bic_score = current_bic_score
                    best_num_components = current_num_states

                if self.verbose:
                    print("model created for {} with {} states".format(self.this_word, current_num_states))

            return self.base_model(best_num_components)
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))

            return self.base_model(num_states)




class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        num_states = 0
        min_dic_score = float("inf")
        best_num_components = 0

        try:
            for current_num_states in range(self.min_n_components, self.max_n_components):
                num_states = current_num_states
                hmm_model = GaussianHMM(current_num_states, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(self.X, self.lengths)

                logL = hmm_model.score(self.X, self.lengths)
                num_data_points = len(self.sequences)

                all_logL_except_current = []
                for key, value in self.sequences:
                    if key != self.this_word:
                        current_other_squence, current_other_lengths = self.all_word_Xlengths[key]
                        except_hmm_model = GaussianHMM(current_num_states, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(current_other_squence, current_other_lengths)

                        current_other_logL = except_hmm_model.score(current_other_squence, current_other_lengths)
                        all_logL_except_current.append(current_other_logL)

                sum_other_logL = np.sum(all_logL_except_current)

                current_dic_score = logL - (1 / len(self.words)) * sum_other_logL

                if current_dic_score < min_dic_score:
                    min_dic_score = current_dic_score
                    best_num_components = current_num_states

                if self.verbose:
                    print("model created for {} with {} states".format(self.this_word, current_num_states))

            return self.base_model(best_num_components)
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))

            return self.base_model(num_states)


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        num_states = 0
        min_avg_logL = float("inf")
        best_num_components = 0

        # if len(self.sequences) < 3:
        #     split_method = KFold(len(self.sequences))
        # else:
        #     split_method = KFold()

        try:
            if len(self.sequences) < 3:
                split_method = KFold(len(self.sequences))
            else:
                split_method = KFold()

            for current_num_states in range(self.min_n_components, self.max_n_components):
                num_states = current_num_states
                current_num_state_scores = []

                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    training_squences_x, training_squences_lengths = combine_sequences(cv_train_idx, self.sequences)

                    test_squences_x, test_squences_lengths = combine_sequences(cv_test_idx, self.sequences)

                    hmm_model = GaussianHMM(current_num_states, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(training_squences_x, training_squences_lengths)

                    logL = hmm_model.score(test_squences_x, test_squences_lengths)

                    current_num_state_scores.append(logL)

                current_avg_logL = np.mean(current_num_state_scores)

                if current_avg_logL < min_avg_logL:
                    min_avg_logL = current_avg_logL
                    best_num_components = current_num_states

                if self.verbose:
                    print("model created for {} with {} states".format(self.this_word, current_num_states))

            return self.base_model(best_num_components)
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))

            return self.base_model(num_states)
            # return self.base_model(best_num_components)

        # except:
        #     if self.verbose:
        #         print("failure on {} with {} states".format(self.this_word, num_states))
        #     return None

