import torch

# +
with open('dataset/wizard_of_oz.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(chars)
print(len(chars))

# +
string_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_string = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])

encoded_hello = encode("Hello")
decoded_hello = decode(encoded_hello)
print(encoded_hello)
print(decoded_hello)
# -

data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000])

n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

block_size = 8
train_data[:block_size+1]

#Time Dimension of tensor
x = train_data[: block_size]
y = train_data[1: block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context} the target is: {target}")

# +
#Batch Size
torch.manual_seed(1337)
batch_size = 4 
block_size = 8

def get_batch(split):
    
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('target:')
print(yb.shape)
print(yb)

print('------')

for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, :t+1]
        target = yb[b,t]
        print(f"When input is {context.tolist()} the target: {target}")
# -

print(xb)

# +
import torch 
import torch.nn as nn 
from torch.nn import functional as F 
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):
    
    def __init__ (self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, targets = None):
        
        logits = self.token_embedding_table(idx) #(B,T,C)
        
        if targets == None: 
            loss = None
            
        else: 
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim = -1)
            idx_next = torch.multinomial(probs, num_samples = 1)
            idx = torch.cat((idx, idx_next), dim = 1)
        return idx
    
m = BigramLanguageModel(vocab_size)
logits, loss = m(xb,yb)
print(logits.shape)
print(loss)
print(decode(m.generate(idx = torch.zeros((1,1), dtype= torch.long), max_new_tokens=100)[0].tolist()))
        
# -

optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3) #pytorch Optimizer

# +
batch_size = 32
for steps in range(10000):
    
    xb, yb = get_batch('train')
    
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
print(loss.item())
# -

print(decode(m.generate(idx = torch.zeros((1,1), dtype= torch.long), max_new_tokens=400)[0].tolist()))

torch.manual_seed(1337)
B,T,C = 4, 8, 2
x = torch.randn(B,T,C)
x.shape

xbow = torch.zeros(B,T,C)
for b in range(B): 
    for t in range(T):
        xprev = x[b,:t+1] 
        xbow[b,t] = torch.mean(xprev, 0)

#version 2
wei = torch.tril(torch.ones(T,T))
wei = wei / wei.sum(1, keepdim=True)
xbow2 = wei @ x #(B,T,T) @ (B,T,C) -> (B,T,C)
torch.allclose(xbow, xbow2) 

# version 3: Uses Softmax
tril = torch.tril(torch.ones(T,T))
wei = torch.zeros((T,T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)
xbow3 = wei @ x 
torch.allclose(xbow, xbow3)

# +
#version 4: Self Attention
torch.manual_seed(1337)
B,T,C = 4, 8, 32
x = torch.randn(B,T,C)

head_size = 16
key = nn.Linear(C, head_size, bias = False)
query = nn.Linear(C, head_size, bias = False)
value = nn.Linear(C, head_size, bias = False)
k = key(x) # (B,T,head_size)
q = query(x) # (B,T,head_size)
wei = q @ k.transpose(-2,-1) # (B,T,head_size) @ (B,head_size,T) -> (B,T,T)

tril = torch.tril(torch.ones(T,T))
#wei = torch.zeros((T,T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)

v = value(x) # (B,T,head_size)
out = wei @ v # (B,T,T) @ (B,T,head_size) -> (B,T,head_size)
#out = wei @ x

out.shape
# -

tril

wei[0]

xbow[0], xbow2[0]

k = torch.randn(B, T, head_size)
q = torch.randn(B, T, head_size)
wei = q @ k.transpose(-2,-1) * head_size**-0.5 # (B,T,head_size) @ (B,head_size,T) -> (B,T,T)

k.var()

q.var()

wei.var()
