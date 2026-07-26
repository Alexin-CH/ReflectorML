import torch
import matplotlib.pyplot as plt

def activation(x, H=1.0):
    T = torch.pi                 # stair width in x is pi
    # map x to stair index n and local coordinate v in [-pi/2, pi/2]
    n = torch.floor((x + torch.pi/2) / T)
    v = (x + torch.pi/2) - n*T - torch.pi/2

    # one stair: from 0 at v=-pi/2 to H at v=pi/2 using sin
    # sin(-pi/2)=-1, sin(pi/2)=1 so (sin(v)+1)/2 maps to [0,1]
    return n*H + H*(torch.sin(v) + 1)/2

xm = 10
x = torch.linspace(-xm, xm, 1000)
y = activation(x).cpu().numpy()

plt.figure()
plt.plot(x, y)
plt.grid(True)
plt.show()
