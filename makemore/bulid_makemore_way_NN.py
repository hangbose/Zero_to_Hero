import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

words = open("names.txt", "r").read().splitlines()
chars = sorted(list(set("".join(words))))
stoi = {s: (i + 1) for i, s in enumerate(chars)}
stoi["."] = 0
itos = {i: s for s, i in stoi.items()}

g = torch.Generator().manual_seed(2147483647)
W = torch.randn(27, 27, generator=g, requires_grad=True)

xs = []
ys = []
num = 0
for w in words:
    chs = "." + w + "."
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)
        num += 1

xs = torch.tensor(xs)
ys = torch.tensor(ys)

xenc = F.one_hot(xs, num_classes=27).float()
print(xenc.shape)

g = torch.Generator().manual_seed(2147483647)

for k in range(300):
    # forward propagate
    logits = xenc @ W
    counts = logits.exp()
    prob = counts / counts.sum(dim=1, keepdim=True)
    loss = -1 * prob[torch.arange(num), ys].log().mean() + 0.01 * (W**2).mean()

    W.grad = None
    loss.backward()
    W.data -= 50 * W.grad


for _ in range(10):
    out = []
    ix = 0
    while True:

        xenc = F.one_hot(
            torch.tensor([ix]), num_classes=27
        ).float()  # [[0, 0, 0, 0, 0, 1, 0, ..., 0]] keep dim 1
        logits = xenc @ W
        counts = logits.exp()
        p = counts / counts.sum(dim=1, keepdim=True, dtype=float)

        ix = (torch.multinomial(p, num_samples=1, replacement=True, generator=g)).item()
        out.append(itos[ix])
        if ix == 0:
            break

    print("".join(out))
