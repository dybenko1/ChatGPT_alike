import torch
import torch.nn as nn
from torch.nn import functional as F
import os


# hyperparameters
batch_size = 32
block_size = 256 
max_iters = 4_000
eval_interval = 500
learning_rate = 6e-4 # self-attention block canoot tolerate high lr
device = 'mps' if torch.mps.is_available() else 'cpu' # cpu because for this small model the mps. option is slower
eval_iters = 200
n_embd = 384 # Dimension of embedding vectors
n_head = 4
n_layer = 3
dropout = 0.2

#-------------


torch.manual_seed(1337)
#!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here all the unique characters that occur in the text
chars = sorted(set(list(text)))
vocab_size = len(chars)
# (embedding) create mapping from characters to integers
stoi = { ch:i for i, ch in enumerate(chars) }
itos = { i:ch for i, ch in enumerate(chars) }

encode = lambda s: [stoi[ch] for ch in s]
decode = lambda l: "".join([itos[i] for i in l]) 

# Train and val splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
    
@torch.no_grad()
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

class Head(nn.Module):
    """ one head of self-attention"""
    # head size is the dimension of vectors where K, Q, V. They should not necessarily live all in the same dim, like in the deepblue3 video
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False) # Linear transformation from embeddding space to the space where head vectors (k,q,v) will live
        self.query = nn.Linear(n_embd, head_size, bias=False) # Since the input is x and when encoded, its embedding (representation) we define it to live in n_embd we perform a linear transformation
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # What does this does? is the prelude of 'wei'?
        # Looks like the above line creates self.tril = ... But why define it in this way?
        # Answer. This last line of code is to create a variable. called 'tril', that is NOT a parameter of the model
        # so Pytorch denominate these type of variables (that are not parameters) "buffer" and we have
        # to assign it to the module with this "register_buffer" function.
        # So that line would be equivalent to "self.tril = torch.tril(torch.ones...)))""

        self.dropout = nn.Dropout(dropout)  # Adding dropout as regularization.

    def forward(self, x):
        # input of size (batch, time-step, channels)
        B, T, C = x.shape 
        q = self.query(x) # (B, T, hs), i.e. Batch size, block size, head dimension (head size)
        k = self.key(x) # (B, T, hs)

        # compute attention scores ('affinities'). **The formula of the paper**
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) --> (B, T, T)
        ##TODO Corregir linea del wei.masked**************************************************************
        wei = wei.masked_fill(self.tril[:T,:T]==0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei) ## Randomly preventing some of the nodes from communicating 

        v = self.value(x) # (B, T, hs). Aunque segun Andrej the dimension should be (B, T, C)

        out = wei @ v #  (B, T, T) @ (B, T, hs) ---> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd) # The pirpose of this projector is to take the output of the attention heads and project them back into the residual pathway. 
        ## It is to map the output (of the heads) back to the data stream (x) dimension, so. there are not dimensionality mismatches; specially useful in situations when 
        ## n_emd != n_heads * head_size
        ## It also allows the model to learn wot to "mix and combine the information" from the different heads together, rather than just treating them as parallel streams of data
        ## In our example there is no dimension msimatch, but the projector is still usefull bceause if we were to change the n_heads or the heads size it would always help to match dimensions
        self.dropout = nn.Dropout(dropout)  # Adding dropout as regularization.

    def forward(self, x):
      out = torch.cat([h(x) for h in self.heads], dim=-1)
      out = self.proj(out)
      return out

class FeedForward(nn.Module):
  def __init__(self, n_embd):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(n_embd, 4 * n_embd),
      nn.ReLU(),
      nn.Linear(4 * n_embd, n_embd), # For the "Residual". This is the projection layer back to the residual pathway. Here we fix a dimension mismatch
      nn.Dropout(dropout), # Adding dropout as regularization. This is added right before the residual conenction  back to the residual pathway
    )

  def forward(self, x): # This linear layer is on token level, i.e. indepently each token "thinks" about the info they retrieve in the past (self-attention layer)
    return self.net(x)

class Block(nn.Module):
  def __init__(self, n_embd, n_head):
    super().__init__()
    head_size = n_embd // n_head
    self.sa = MultiHeadAttention(n_head, head_size)
    self.ffwd = FeedForward(n_embd)
    self.ln1 = nn.LayerNorm(n_embd) ## Andrej showed the equivalent function of this function that come from a paper. And in here we deviate from the original "Attention is all you need" paper and apply it before, not after the attention heads
    ## Ojo con las dimensiones. Aca la operacion de nromalziacion (mean of 1 and sd of 1) happens over 32 numbers (n_embd), so the batch and time as a batch dimen, both of them
    ## So this is like a "per token" transformation. It normalizes the features at inizialization, but since therformula has the gamma and beta parameters, later the
    ## norm Layer norm would produce outputs that may not be "standard" (meand 1 and sd 1)
    ## OJO: I explain the importance of this layer in the commit.
    self.ln2 = nn.LayerNorm(n_embd)

  def forward(self, x):
    x = x + self.sa(self.ln1(x)) # This is the residual part. We add the "x + ". Here we then added the Layer Norm 1
    x = x + self.ffwd(self.ln2(x)) # This is the residual part. We add the "x + ". Here we then added the Layer Norm 2
    return x



# simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a look up table
        self.token_embdding_table = nn.Embedding(vocab_size, n_embd) # n_embd: number of embedding directions
        # to go from the token embeddings to the logits we need a linear layer
        # We do not just need to encode the tokens given their identity (the word/meaning itself), but also the position
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # each position from 0 to block_size -1 will get its own embedding vector
        # Adding blocks (transformers)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)]) # Changing the format of the block 
        self.ln_f = nn.LayerNorm(n_embd) # Putting a final Layer Norm
        #self.sa_heads = MultiHeadAttention(4, n_embd//4) # i.e. 4 heads of 8-dimensional self-attention. # Interesting because with a single head the dim was 32,

        #self.ffwd = FeedForward(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size) # lm_head : language model head. !! Why this dimension?
        ## OjO! we already abandoned the representation of tokens as integers. we now represent them as a vector of 32 dimensions
        ## that's the reason of n_embd=32. With this many directions (dims) each token can hold complex rich 
        ## meanings given the relationships with the other tokens (words).
        ## It is relevant to notice that we are still in the Bigram model, i.e. only a single token used
        ## to predict the "next" one. So for a single token we need to, again, get the probabilities of the 
        ## occurrence of each of the next tokens; like in the token_embedding_table.
        ## but now instead of hat table we will use a linear transformation to go from
        ## the token space (32 dim) into the vocab_size (the total numbers of characters). 
        ## So long story short: this linear layer will give us the probabilities of occurrence of any of the 
        ## vocabulary given a (input) token.

        

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B, T) tensor of integers
        tok_emb = self.token_embdding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(0, T, device=device)) # (T,C) From zero to T-1
        x = tok_emb + pos_emb # (B,T,C). pos_em no tiene dim B (batch), pero pytorch hace su magia al sumar eso a cada 
        ## batch. at the end the embedding of position i is the same regardless the batch. We just add that to our vector of n_embd dimensions

        #x = self.sa_heads(x) # apply the multi-head attention
        # Forward through transformer blocks 
        x = self.blocks(x)
        # When we developed the multi-head attention head we right away calculated the logits. This did not allow
        # the tokens to "think" what they find about the other tokens (with the multi-head attention), this is why we implement
        # a linear layer for the tokens to "grasp" what they found when they communicate with each other (in the self-attention layer)
        #x = self.ffwd(x) # (B,T, C)
        logits = self.lm_head(x) # (B, T, vocab_size). 
        

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            loss = F.cross_entropy(logits, targets)
        
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the prediction
            logits, loss = self(idx_cond)
            # We grab the logits of the last token (the only useful to predict)
            logits = logits[:, -1, :] # becomes (B, C). 
            # apply softmax to get probabilities 
            probs = F.softmax(logits, dim=1) # (B, C)
            # sample from distribution 
            idx_next = torch.multinomial(probs,1) # (B, 1)
            # append sampled index to the running sequence 
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    
model = BigramLanguageModel()

# In case model was pre-trained we just load weights
weights = "SavedWeights/model_weight.pth"
if os.path.exists(weights) and os.path.isfile(weights):
        # Load pre-saved model's weights
   model.load_state_dict(torch.load(weights, map_location=device))
else: # Train the model...
    print("HOLAXXXXX")
    m = model.to(device)
    # create a Pytorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Saving model after training
    torch.save(model.state_dict(), 'model_weight.pth')
    torch.save(model.state_dict(), 'SavedWeights/model_weight.pth')
    print("Model saved to model_weights.pth")

m = model.to(device)
m.eval() # Ultra relevant to turn off the "dropout" feature (that randomly ignores certain neurons, only useful for training as a regularization technique)

# generate from the model
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))



