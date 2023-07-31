from datasets import load_dataset
import re
from tqdm import tqdm
import string

def get_allowed_chars():
    print('Getting Valid Characters...')
    allowed_chars = [char for char in tqdm(string.ascii_letters + string.digits + string.punctuation + ' ' + '\n', desc = 'Getting Valid Characters')]
    return allowed_chars

def filter_data(data, allowed_chars):
    filtered_data = []
    for text in tqdm(data, desc='Filtering Data'):
        filtered_text = ''.join(c for c in text if c in allowed_chars)
        filtered_data.append(filtered_text)
    return filtered_data

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

def get_vocab(data_list):
    words = []
    for i in tqdm(data_list, desc='Getting Vocabulary'):
        words += i.split(' ')

    vocab = [*set(words)]
    vocab = remove_mask_oov_tokens(vocab)

    return vocab


print('Loading Dataset...')
#load_dataset("wikipedia", "20220301.en") # this is the huge dataset with over 20G of storage!!
full_dataset = load_dataset('wikipedia', '20220301.simple') # this one has only 235M of storage - use for fast n easy! 

dataset = list(full_dataset['train'])

full_data = [i['text'] for i in dataset]

chars = get_allowed_chars()
print('Filtering Data...')
data = filter_data(full_data, chars)
print('Getting Vocabulary...')
vocab = get_vocab(data)
vocab_size = len(vocab)
chars_size = len(chars)

'''with open('variables.pickle', 'wb') as file:
    pickle.dump(chars, file)
    pickle.dump(data, file)
    pickle.dump(vocab, file)
    pickle.dump(vocab_size, file)
    pickle.dump(chars_size, file)'''

print('Variables saved successfully.')