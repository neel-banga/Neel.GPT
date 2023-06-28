from datasets import load_dataset
import tensorflow as tf
import pickle
import re
import os

def remove_mask_oov_tokens(vocabulary):
    mask_token = ''
    oov_token = '[UNK]'
    cleaned_vocabulary = [token for token in vocabulary if token not in [mask_token, oov_token]]
    return cleaned_vocabulary

def remove_special_chars(string):
    string = re.sub(r'[.,]', '', string)
    string = re.sub(r'[\'\"]', '', string)
    string = re.sub(r'[()]', '', string)
    string = re.sub(r'[\[\]]', '', string)
    return string

def get_vocab(list):
    words = []
    for i in list:
        words += i.split(' ')

    vocab = [*set(words)]
    vocab = remove_mask_oov_tokens(vocab)

    return vocab

def train_vectorization_model(vocab):
    vectorize_layer = tf.keras.layers.TextVectorization(
        max_tokens=None,
        standardize='lower_and_strip_punctuation',
        split='whitespace',
        ngrams=None,
        output_mode='int',
        output_sequence_length=None,
        pad_to_max_tokens=False,
        vocabulary=vocab,
    )

    pickle.dump({'config': vectorize_layer.get_config(),
                'weights': vectorize_layer.get_weights()}
                , open('vectorization_layer.pkl', 'wb'))

#load_dataset("wikipedia", "20220301.en") # this is the huge dataset with over 20G of storage!!
full_dataset = load_dataset('wikipedia', '20220301.simple') # this one has only 235M of storage - use for fast n easy! 

# https://huggingface.co/datasets/wikipedia
# https://dumps.wikimedia.org/backup-index.html

dataset = list(full_dataset['train'])

#print(dataset[0]) # -> dictionary of 'id', 'url', 'title', and 'text' - we only need text and possibly title

data = [i['text'] for i in dataset]

vocab = get_vocab(data)
print('dun dun dun got vocab')


if not os.path.isfile('vectorization_layer.pkl'):
    train_vectorization_model(vocab)


from_disk = pickle.load(open('vectorization_layer.pkl', 'rb'))
vectorizer = tf.keras.layers.TextVectorization.from_config(from_disk['config'])
vectorizer.set_weights(from_disk['weights'])

print('dun dun dun vectorized')

# This dataset only has training data provided so let's fix that
get_split_index = lambda x, list: int(len(list) * x/100)

training_data = dataset[:get_split_index(80, dataset)]
validation_data = dataset[:get_split_index(10, dataset)]
testing_data = dataset[:get_split_index(10, dataset)]

#output = vectorizer(dataset[0]['text'])
#print(output)