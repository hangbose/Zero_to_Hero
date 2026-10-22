import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

words = open("names.txt", "r").read().splitlines()

N = torch.zeros((27, 27), dtype=torch.int32)

chars = sorted(list(set("".join(words))))

stoi = {s: (i + 1) for i, s in enumerate(chars)}
stoi["."] = 0
itos = {i: s for s, i in stoi.items()}

for w in words:
    chs = "." + w + "."
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1


plt.figure(figsize=(16, 16))
plt.imshow(N, cmap="Blues")

for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]

        # 显示字符组合，例如 ab、em、..
        plt.text(j, i, chstr, ha="center", va="bottom", color="gray")

        # 显示出现次数
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color="gray")

plt.axis("off")

g = torch.Generator().manual_seed(2147483647)


P = (N + 1).float()  # +1 smooth the result
# 27 27  size([27, 27])
"""[ 
     [  ]
     [  ]
     [  ]
     [  ]   
    ]
"""
P /= P.sum(dim=1, keepdim=True)
log_likelihood = 0.0
n = 0
#loss function
for w in words[:3]:
    chs = "." + w + "."
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1][ix2]
        logprob = torch.log(prob)
        log_likelihood += logprob  #p(.) * p(char)* ..   
        ##print(f"{ch1}{ch2}: {prob} {logprob}")
        n += 1
print(f"{log_likelihood=}")
nll = -log_likelihood
print(f"{nll=}")
print(f"{nll/n=}")

#the trained model predict the words 
print(P[0])
for _ in range(10):
    out = []
    ix = 0
    while True:
        p = P[ix]
        ix = (torch.multinomial(p, num_samples=1, replacement=True, generator=g)).item()
        out.append(itos[ix])
        if ix == 0:
            break

    print("".join(out))


