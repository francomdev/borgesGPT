import torch
import torch.nn as nn
from torch.nn import functional as F
import time

#Hyperparameters
block_size = 256
batch_size = 64
eval_iters = 200
eval_interval = 500
epochs = 5000
n_embd = 384
n_head = 6
n_layer = 6
head_size = n_embd/n_head
dropout = 0.2
device = 'cpu'

TRAIN = False

#Import dataset
with open ('borges.txt', 'r', encoding='utf-8') as f:
    text = f.read()
print(len(text))
#Get all the characters in order
chars = sorted(list(set(text)))
vocab_size = len(chars)
#Tokenize in a character level
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] #Turns text into integers
decode = lambda l: ''.join([itos[i] for i in l]) # Inverse operation
#Pack encoded text in a tensor
data = torch.tensor(encode(text), dtype=torch.long)

#Split train and validation sets
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]
#Defining a block of data
#block_size = 8 # --> Maximum context length
# x = train_data[:block_size]
# y = train_data[1:block_size+1]
# for t in range(block_size):
#     context = x[:t+1]
#     target = y[t]
#     print(f'When the context is {context}, the target is {target}')
#Creating batches
torch.manual_seed(1337)


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])    
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y 

@torch.inference_mode()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
            out[split] = losses.mean()
    model.train()
    return out


xb, yb = get_batch('train')
#We will implement a bigram model, the simplest one, and see how it does.

class Head(nn.Module):
    '''
    Single head of self attention
    '''
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5 #Compute attention scores
        wei = wei.masked_fill(self.tril[:T, :T] == 0 , float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)   
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    '''
    a simple linear layer followed by a non linearity
    '''
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd), 
            nn.Dropout(dropout)
        )
    def forward(self,x):
        return self.net(x)

class Block(nn.Module):
    '''
    Transformer block: communication followed by computation
    '''
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        #Each token reads the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer normalization
        self.lm_head = nn.Linear(n_embd, vocab_size)
    def forward(self, idx, targets = None): 
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) 
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) #(B, T, C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # This is because of how PyTorch expects cross entropy inputs
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        #idx is (B,T) the current context
        for _ in range(max_new_tokens):
            #get the predictions
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            #focus only on last time frame
            logits = logits[:,-1,:]
            probs = F.softmax(logits, dim=1)
            idx_next = torch.multinomial(probs, num_samples=1) #Sample a new character with the obtained distribution
            #append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) #(B,T+1)
        return idx

model = BigramLanguageModel().to(device)
logits, loss = model(xb, yb)

#We generate some text with the random model
print("Random model generation:")
random_text = decode(model.generate(torch.zeros((1,1), dtype=torch.long, device=device), max_new_tokens=500)[0].tolist())

print(random_text)
print("")
#Now we train the model

optimizer = torch.optim.AdamW(model.parameters(), lr = 3e-4)

if TRAIN:

    for epoch in range(epochs):
        if epoch % eval_interval == 0:
            losses = estimate_loss()
            print(f"Epoch: {epoch} | train_loss: {losses['train']} | val loss: {losses['val']}")
        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

else:
    model.load_state_dict(torch.load('model_params', map_location=device))

#We generate some text with the trained model
print("Trained model generation")
output_text = decode(model.generate(torch.zeros((1,1), dtype=torch.long, device=device), max_new_tokens=1000)[0].tolist())


def print_text_animated(text):
    for char in text:
        print(char, end = "", flush=True)
        time.sleep(0.02)

print_text_animated(output_text)