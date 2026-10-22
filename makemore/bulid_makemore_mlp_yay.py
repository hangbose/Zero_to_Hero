import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

words = open("names.txt", "r").read().splitlines()
chars = sorted(list(set("".join(words))))
stoi = {s: (i + 1) for i, s in enumerate(chars)}
stoi["."] = 0
itos = {i: s for s, i in stoi.items()}

xs = []
ys = []
block_size = 5
context = block_size * [0]

for w in words[:3]:
    for ch in w + '.':
        ix = stoi[ch]
        xs.append(context)
        ys.append(ix)
        print(f"{''.join(itos[i] for i in context)} ------> {ch}")
        context = context[1:] + [ix]

xs = torch.tensor(xs)
ys = torch.tensor(ys)

