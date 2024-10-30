import torch
from torch.autograd.functional import jacobian
import numpy as np
def compute_jacobian(model, inputs):
    """Computes jacobian of `output` w.r.t. `inputs`.
    Args:
        model: A PyTorch model.
        inputs: A tensor or a nested structure of tensor objects.
    Returns:
        A tensor or a nested structure of tensors with the same structure as
        `inputs`. Each entry is the jacobian of `output` w.r.t. to the corresponding
        value in `inputs`. If output has shape [y_1, ..., y_n] and inputs_i has
        shape [x_1, ..., x_m], the corresponding jacobian has shape
        [y_1, ..., y_n, x_1, ..., x_m].
    """
    num_params = sum(p.numel() for p in model.parameters())
    inputs = torch.tensor(inputs.astype(np.float32))
    gradients = np.zeros((len(inputs), num_params))
    for i in range(len(inputs)):
        #model.zero_grad()
        input_tensor = torch.tensor(inputs[i], dtype=torch.float32).unsqueeze(0)  # 1x10
        output = model(input_tensor)
        output.backward()
        grads = []
        for param in model.parameters():
            grads.append(param.grad.view(-1).detach().numpy())
        gradients[i, :] = np.concatenate(grads)    
    return gradients
