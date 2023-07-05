import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from models import Model
    

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f'Using {device} device')


def computeGrads(y,x):
    grads, = torch.autograd.grad(y,x,grad_outputs=torch.ones(y.shape), create_graph=True)
    return grads

max_epochs = int(input('Max number of iterations: '))
n_samples = 1000
n_samples_init = 1000
log_each = int(input('Log each: '))

model = Model()
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[int(0.4*max_epochs), int(0.8*max_epochs)], gamma=0.1)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=int(0.05*max_epochs), gamma=0.8,last_epoch=int(0.9*max_epochs))
#scheduler3 = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer,gamma=0.999,last_epoch=int(0.1*max_epochs))
loss_fn = torch.nn.MSELoss()
model.train()
model.to(device)
Re = 100

pde_losses, bound_losses = [], []

for epoch in range(max_epochs + 1):
    # PDE
    X = torch.rand((n_samples,2), requires_grad=True, device=device)
    Y_hat = model(X)
    u, v, p = Y_hat[:,0], Y_hat[:,1], Y_hat[:,2]
    
    du, dv, dp = computeGrads(u,X), computeGrads(v,X), computeGrads(p,X)
    dudx, dudy = du[:,0], du[:,1]
    dvdx, dvdy = dv[:,0], dv[:,1]
    dpdx, dpdy = dp[:,0], dp[:,1]
    
    du2dx2, du2dy2 = computeGrads(dudx, X)[:, 0], computeGrads(dudy, X)[:, 1]
    dv2dx2, dv2dy2 = computeGrads(dvdx, X)[:, 0], computeGrads(dvdy, X)[:, 1]
    
    pde_loss = loss_fn(dudx, - dvdy) + \
        loss_fn(u*dudx + v*dudy + dpdx, (1./Re)*(du2dx2 + du2dy2)) + \
        loss_fn(u*dvdx + v*dvdy + dpdy, (1./Re)*(dv2dx2 + dv2dy2))
    pde_losses.append(pde_loss.item())
    
    #############################################
    # Boundary conditions
    #############################################
    
    # Left
    Y0 = torch.stack([
        torch.zeros(n_samples_init, device=device),
        torch.rand(n_samples_init, device=device),
    ], axis=-1)
    Y0.requires_grad = True
    p_y0 = torch.stack([
        torch.zeros(n_samples_init, device=device),
        torch.zeros(n_samples_init, device=device),
    ], axis=-1)
    y_y0 = model(Y0)
    y0_uv_loss = loss_fn(y_y0[:,:2], p_y0)
    p = y_y0[:,2]
    dpdx = computeGrads(p, Y0)[:,0]
    y0_p_loss = loss_fn(dpdx, torch.zeros(len(dpdx), device=device))
    
    # Rigth
    Y1 = torch.stack([
        torch.ones(n_samples_init, device=device),
        torch.rand(n_samples_init, device=device),
    ], axis=-1)
    Y1.requires_grad = True
    p_y1 = torch.stack([
        torch.zeros(n_samples_init, device=device),
        torch.zeros(n_samples_init, device=device),
    ], axis=-1)
    y_y1 = model(Y1)
    y1_uv_loss = loss_fn(y_y1[:,:2], p_y1)
    p = y_y1[:,2]
    dpdx = computeGrads(p, Y1)[:,0]
    y1_p_loss = loss_fn(dpdx, torch.zeros(len(dpdx), device=device))
    
    # Bottom
    X0 = torch.stack([
        torch.rand(n_samples_init, device=device),
        torch.zeros(n_samples_init, device=device),
    ], axis=-1)
    X0.requires_grad = True
    p_x0 = torch.stack([
        torch.zeros(n_samples_init, device=device),
        torch.zeros(n_samples_init, device=device),
    ], axis=-1)
    y_x0 = model(X0)
    x0_uv_loss = loss_fn(y_x0[:,:2], p_x0)
    p = y_x0[:,2]
    dpdy = computeGrads(p, X0)[:,1]
    x0_p_loss = loss_fn(dpdy, torch.zeros(len(dpdy), device=device))
    
    # Top
    X1 = torch.stack([
        torch.rand(n_samples_init, device=device),
        torch.ones(n_samples_init, device=device),
    ], axis=-1)
    X1.requires_grad = True
    p_x1 = torch.stack([
        torch.ones(n_samples_init, device=device),
        torch.zeros(n_samples_init, device=device),
    ], axis=-1)
    y_x1 = model(X1)
    x1_uv_loss = loss_fn(y_x1[:,:2], p_x1)
    p = y_x1[:,2]
    dpdy = computeGrads(p, X1)[:,1]
    x1_p_loss = loss_fn(dpdy, torch.zeros(len(dpdy), device=device))
    
    bound_loss = y0_uv_loss + y0_p_loss + \
        x1_uv_loss + x1_p_loss + \
        x0_uv_loss + x0_p_loss + \
        y1_uv_loss + y1_p_loss
        
    bound_losses.append(bound_loss.item())
    
    if epoch % log_each == 0:
        print(f'Epochs: {epoch}/{max_epochs} | loss (PDE): {pde_loss:.2e} | loss (bcs): {bound_loss:.2e} | learning rate: {scheduler1.get_last_lr()[0]:.2e}')
        
    # Backpropagation
    optimizer.zero_grad()
    loss = 1.*bound_loss + pde_loss
    
    if pde_loss <= 1e-4 and bound_loss <= 1e-4:
        break
    
    loss.backward()
    optimizer.step()
    scheduler1.step()
    scheduler2.step()
    #scheduler3.step()


save_model = input('Save Model? (yes/no): ')
if save_model.lower() == 'yes':
    model_name = input('Model name: ')
    torch.save(model, f'{model_name}.pth')
    
    losses_df = pd.DataFrame({
        'Epochs': np.arange(len(pde_losses)),
        'PDE Loss': pde_losses,
        'Boundary Conditions Loss': bound_losses
    })

    losses_df.to_csv(f'Losses-{model_name}.csv', index=False)


plt.plot(pde_losses,'.-', label='PDE')
plt.plot(bound_losses, '.-', label='Boundary Conditions')
plt.yscale('log')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()