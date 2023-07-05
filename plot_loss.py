import sys
import os

import pandas as pd
import matplotlib.pyplot as plt

filename = sys.argv[1]
data_df = pd.read_csv(filename)

pde_loss = data_df['PDE Loss']
bcs_loss = data_df['Boundary Conditions Loss']

plt.plot(pde_loss,'.-', label='PDE')
plt.plot(bcs_loss, '.-', label='Boundary Conditions')
plt.yscale('log')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()

save_request = input('save (yes/no): ')
if save_request.lower() == 'yes':
    filename = input('filename: ')
    plt.savefig(filename, dpi=1000)

plt.show()