import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    for idx in range(0, test_set.num_items):
        current_frame_dict = {}
        highest_key = ''
        highest_value = float('-inf')

        for model_key, model_value in models.items():
            try:
                current_sequences, current_lengths = test_set.get_item_Xlengths(idx)
                current_logL = model_value.score(current_sequences, current_lengths)
                current_frame_dict[model_key] = current_logL

                if current_logL > highest_value:
                    highest_value = current_logL
                    highest_key = model_key

            except:
                continue

        probabilities.append(current_frame_dict)
        guesses.append(highest_key)

    return probabilities, guesses