import torch
import concurrent.futures
from tqdm import tqdm
#from data import data, vocab, chars, vocab_size, chars_size

SPLIT_VAL = 10 # Decrease to 1 once you have GPU
BLOCK_SIZE = 8


# Loading variables
try:
    import pickle

    with open('variables.pickle', 'rb') as file:
        chars = pickle.load(file)
        data = pickle.load(file)
        vocab = pickle.load(file)
        vocab_size = pickle.load(file)
        chars_size = pickle.load(file)

    print('Variables loaded successfully. \n')

except:
    from data import data, vocab, chars, vocab_size, chars_size

def split_list(lst, num_chunks):
    chunk_size = len(lst) // num_chunks
    remainder = len(lst) % num_chunks
    chunks = []
    index = 0

    for _ in tqdm(range(num_chunks), desc = 'Splitting List'):
        if remainder > 0:
            chunk = lst[index:index+chunk_size+1]
            index += chunk_size + 1
            remainder -= 1
        else:
            chunk = lst[index:index+chunk_size]
            index += chunk_size
        chunks.append(chunk)
    
    return chunks


def encode(sentance):
    values = []
    for char in sentance:
        values.append(chars.index(char))

    return values

def decode(tokens):
    values = []
    for token in tokens:
        values.append(chars[token])
    print(values)
    return ''.join(values)


'''tokens = encode('hi! there')
print(tokens)
translation = decode(tokens)
print(translation)
'''

def encode_data(data, split):
    # Third data to make it easier to work with (can take this step back once I have a gpu)
    data = split_list(data, split)[0]


    #encoded_data = [torch.tensor(encode(block)) for block in data]
    #print(encoded_data.shape())

    num_threads = 16
    executor = concurrent.futures.ThreadPoolExecutor(num_threads)
    print('Starting to encode')
    encoded_data = list(tqdm(executor.map(encode, data), desc = 'Encoding'))
    executor.shutdown()

    print('Now To Tensor')

    encoded_data = [torch.tensor(i) for i in encoded_data]
    print('Encoded')

    return encoded_data


'''encoded_data = encode_data(data, SPLIT_VAL)
print('ed')
print(encoded_data)'''

try:
    with open('data.pickle', 'rb') as file:
        encoded_data = pickle.load(file)
        train_data = pickle.load(file)
        val_data = pickle.load(file)

except:
    encoded_data = encode_data(data, SPLIT_VAL)
    split_line = int(0.9*len(encoded_data))
    train_data = encoded_data[:split_line]
    val_data = encoded_data[split_line:]


    with open('data.pickle', 'wb') as file:
        pickle.dump(encoded_data, file)
        pickle.dump(train_data, file)
        pickle.dump(val_data, file)

block = train_data[0][:BLOCK_SIZE+1]
targets = train_data[0][1:BLOCK_SIZE+2]

for i in range(len(block)):
    print(decode(block[:i+1].numpy().tolist()))
    

    print(f'Context {block[:i+1]}') # Non inclusive that's why we have +1 because if we do [:0] it'll get all elements UNTIL 0
    print(f'Target {targets[i]}')