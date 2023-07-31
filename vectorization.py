import torch
from tqdm import tqdm
from data import data, vocab, chars, vocab_size, chars_size

SPLIT_VAL = 20 # Decrease to 1 once you have GPU
BLOCK_SIZE = 8

def split_list(lst, num_chunks = SPLIT_VAL):
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
    
    return chunks[0]

def encode(sentence):
    values = []
    for char in sentence:
        values.append(chars.index(char))

    return values

def decode(tokens):
    values = []
    for token in tokens:
        values.append(chars[token])
    print(values)
    return ''.join(values)

data = split_list(data, SPLIT_VAL)


print(data[0])

string_data = ''.join(data)

encoded_data = torch.tensor(encode(string_data))
print(encoded_data.shape)

split_line = int(0.9*len(encoded_data))
train_data = encoded_data[:split_line]
val_data = encoded_data[split_line:]

block = train_data[:BLOCK_SIZE+1]
targets = train_data[1:BLOCK_SIZE+2]

for i in range(len(block)):
    print(decode(block[:i+1].numpy().tolist()))
    
    print(f'Context {block[:i+1]}') # Non inclusive that's why we have +1 because if we do [:0] it'll get all elements UNTIL 0
    print(f'Target {targets[i]}')

torch.manual_seed(123)
batch_size = 4
block_size = 8

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

print(train_data)

x_batch, y_batch = get_batch('train')
print('inputs:')
print(x_batch.shape)
print(x_batch)
print('targets:')
print(y_batch.shape)
print(y_batch)

print('----')

for b in range(batch_size):
    for t in range(block_size):
        context = x_batch[b, :t+1]
        target = y_batch[b,t]
        print(f"when input is {context.tolist()} the target: {target}")