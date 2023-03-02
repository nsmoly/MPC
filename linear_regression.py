
# (c) 2022 Nikolai Smolyanskiy
# This code is licensed under MIT license (see LICENSE.txt). No warranties

# This code demonstrates linear regression (as test) using PyTorch for optimization. 

import torch
import numpy as np
import matplotlib.pyplot as plt

max_count = 10

a = 2.0
b = 5.0
x = [float(x_i) for x_i in range(0, max_count) ]
n = np.random.randn(max_count)
y = [a*x[i] + b + n[i] for i in range(0, max_count)]
y_line = [a*x[i] + b for i in range(0, max_count)]
plt.axes().set_aspect('equal', 'datalim')
plt.plot(x, y, 'bo')
plt.plot(x, y_line, 'g')
plt.show()

a_param = torch.tensor(0.1, requires_grad=True)
b_param = torch.tensor(-0.1, requires_grad=True)
optim = torch.optim.AdamW([a_param, b_param], lr=0.5)

for iter in range(0, 100):
    print(f"a={a_param.item()}, b={b_param.item()}")
    optim.zero_grad()

    x_tl = [torch.tensor(x[i]) for i in range(0,max_count)]
    y_target_tl = [torch.tensor(y[i]) for i in range(0,max_count)]
    
    y_pred_tl = [a_param*x_tl[i] + b_param for i in range(0,max_count)]
    errors = [(y_target_tl[i]-y_pred_tl[i])**2.0 for i in range(0, max_count)]
    loss = torch.sum(torch.stack(errors, dim=0), dim=0)
    loss.backward()
    
    optim.step()
    print(f"Iteration:{iter}, loss:{loss.item()}")

y_fit_line = [a_param.data*x[i] + b_param.data for i in range(0, max_count)]
plt.axes().set_aspect('equal', 'datalim')
plt.plot(x, y, 'bo')
plt.plot(x, y_fit_line, 'r')
plt.show()



