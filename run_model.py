import sys

import torch
import torch.nn
import numpy as np
import matplotlib.pyplot as plt
from models import Model, Sine


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


model_file = sys.argv[1]
print(model_file)
model = torch.load(model_file)


def run_mlp(Nx, Ny):
    x = np.linspace(0,1,Nx)
    y = np.linspace(0,1,Ny)
    X = np.stack(np.meshgrid(x,y), -1).reshape(-1, 2)
    X = torch.from_numpy(X).float()
    model.eval()
    model.cpu()
    with torch.no_grad():
        p = model(X)
    return p[:,0].reshape(Ny,Nx), p[:,1].reshape(Ny,Nx), p[:,2].reshape(Ny,Nx), x, y

Nx, Ny = 2000, 2000
u, v, p, x, y = run_mlp(Nx, Ny)

# plot results
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10,8))
vel = np.sqrt(u**2 + v**2)
im=ax1.imshow(vel, vmin=0, vmax=1, origin='lower', extent=[x.min(), x.max(), y.min(), y.max()])
fig.colorbar(im, ax=ax1)
ax1.set_xlabel("x", fontsize=14)
ax1.set_ylabel("y", fontsize=14, rotation=np.pi/2)
ax1.set_title("Speed ($\mathbf{u}$)")
ax1.axis(False)
im=ax2.imshow(u, vmin=u.min(), vmax=u.max(), origin='lower', extent=[x.min(), x.max(), y.min(), y.max()])
fig.colorbar(im, ax=ax2)
ax2.set_title("Velocity-X ($u_x$)")
im=ax3.imshow(v, vmin=v.min(), vmax=v.max(), origin='lower', extent=[x.min(), x.max(), y.min(), y.max()])
fig.colorbar(im, ax=ax3)
ax2.axis(False)
ax3.axis(False)
ax3.set_title("Velocity-Y ($u_y$)")
im=ax4.imshow(p, vmin=p.min(), vmax=p.max(), origin='lower', extent=[x.min(), x.max(), y.min(), y.max()])
fig.colorbar(im, ax=ax4)
ax4.axis(False)
ax4.set_title("Preasure ($p$)")
plt.tight_layout()

save_request = input('save (yes/no): ')
if save_request.lower() == 'yes':
    filename = input('filename: ')
    plt.savefig(filename, dpi=1000)

plt.show()