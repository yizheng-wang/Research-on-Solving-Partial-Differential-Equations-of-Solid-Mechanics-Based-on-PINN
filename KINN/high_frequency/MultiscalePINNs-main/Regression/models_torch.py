import torch
import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as F
import numpy as np
from Compute_Jacobian_torch import compute_jacobian
import time
import math
# Data Sampler
class Sampler:
    # Initialize the class
    def __init__(self, dim, coords, func, name=None):
        self.dim = dim
        self.coords = coords
        self.func = func
        self.name = name

    # Sample function
    def sample(self, N):
        x = self.coords[0:1, :] + (self.coords[1:2, :] - self.coords[0:1, :]) * np.random.rand(N, self.dim)
        y = self.func(x)
        return x, y
    

class NN_FF(nn.Module):
    def __init__(self, layers, X_u, Y_u, a, u, sigma):
        super(NN_FF, self).__init__()

        """
        :param layers: Layers of the network
        :param X_u, Y_u: Training data
        :param a:  Hyper-parameter of the target function
        :param u:  the target function

        """
        
        self.mu_X, self.sigma_X = X_u.mean(0), X_u.std(0)
        self.mu_x, self.sigma_x = self.mu_X[0], self.sigma_X[0]

        # Normalize the input of the network
        self.X_u = X_u 
        self.X_u_tensor = torch.tensor(self.X_u.astype(np.float32))
        self.Y_u = Y_u
        self.Y_u_tensor = torch.tensor(self.Y_u.astype(np.float32))
        # Initialize Fourier features
        # self.W = nn.Parameter(torch.randn(1, layers[0] // 2) * sigma, requires_grad=False)

        # Initialize network layers
        
        # Define the size of the Kernel
        self.D_u = X_u.shape[0]
        
        # Placeholder for NTK matrix
        self.K = None

        D_in = layers[0]
        H = layers[1]
        D_out = layers[2]
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, H)
        self.linear4 = torch.nn.Linear(H, H)
        self.linear5 = torch.nn.Linear(H, D_out)

        torch.nn.init.normal_(self.linear1.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear2.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear3.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear4.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear5.bias, mean=0, std=1)

        torch.nn.init.normal_(self.linear1.weight, mean=0, std=np.sqrt(2/(D_in+H)))
        torch.nn.init.normal_(self.linear2.weight, mean=0, std=np.sqrt(2/(H+H)))
        torch.nn.init.normal_(self.linear3.weight, mean=0, std=np.sqrt(2/(H+H)))
        torch.nn.init.normal_(self.linear4.weight, mean=0, std=np.sqrt(2/(H+H)))
        torch.nn.init.normal_(self.linear5.weight, mean=0, std=np.sqrt(2/(H+D_out)))

    # Initialize network layers using Xavier initialization
    def initialize_NN(self, layers):
        layers_list = []
        num_layers = len(layers)
        for l in range(num_layers - 1):
            layers_list.append(nn.Linear(layers[l], layers[l + 1]))
            if l != num_layers - 2:
                layers_list.append(nn.Tanh())
        return nn.Sequential(*layers_list)
    
    # Neural network forward pass
    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        yt = x
        y1 = torch.tanh(self.linear1(yt))
        y2 = torch.tanh(self.linear2(y1))
        y3 = torch.tanh(self.linear3(y2)) + y1
        y4 = torch.tanh(self.linear4(y3)) + y2
        y =  self.linear5(y4)
        return y

    # Compute NTK
    def compute_NTK(self):
        G = compute_jacobian(self, self.X_u)
        K = result = np.dot(G, G.T)
        return K

    # Train the model
    def train_model(self, nIter, log_NTK=False, log_weights=False):
        self.loss_u_log = []
        self.train_error_log = []
        self.test_error_log = []
        self.K_log = []
        self.weights_log = []
        self.biases_log = []

        optimizer = optim.Adam(self.parameters(), lr = 0.001)
        criterion = nn.MSELoss()
        def closure():
            optimizer.zero_grad()
            u_pred = self.forward(self.X_u_tensor)
            loss = criterion(u_pred, self.Y_u_tensor)
            loss.backward()
            return loss
        start_time = time.time()
        for it in range(nIter):
            loss_value = optimizer.step(closure)
            self.loss_u_log.append(loss_value.data)
            if it % 10 == 0:
                elapsed = time.time() - start_time
                print('It: %d, Loss: %.3e, Time: %.2f' %
                      (it, loss_value.item(), elapsed))

                start_time = time.time()

            # Store the NTK matrix for every 100 iterations
            if log_NTK and it % 100 == 0:
                print("Compute NTK...")
                self.K = self.compute_NTK()
                self.K_log.append(self.K)

            # Store the weights and biases of the network for every 100 iterations
            if log_weights and it % 100 == 0:
                print("Weights stored...")
                self.weights_log.append([p.data.clone() for p in self.parameters()])
                self.biases_log.append([p.data.clone() for p in self.parameters() if p.requires_grad])

    # Evaluate predictions at test points
    def predict_u(self, X_star):
        with torch.no_grad():
            u_star = self.forward(torch.tensor(X_star, dtype=torch.float32))
        u_pred_numpy = u_star.numpy().astype(np.float64)
        return u_pred_numpy


class NN_KAN(nn.Module):
    def __init__(self, layers_hidden,  X_u, Y_u, a, u, sigma, grid_size=10, spline_order=3, grid_range=[-1,1]):
        super(NN_KAN, self).__init__()

        """
        :param layers: Layers of the network
        :param X_u, Y_u: Training data
        :param a:  Hyper-parameter of the target function
        :param u:  the target function

        """
        
        self.mu_X, self.sigma_X = X_u.mean(0), X_u.std(0)
        self.mu_x, self.sigma_x = self.mu_X[0], self.sigma_X[0]

        # Normalize the input of the network
        self.X_u = X_u
        self.X_u_tensor = torch.tensor(self.X_u.astype(np.float32))
        self.Y_u = Y_u
        self.Y_u_tensor = torch.tensor(self.Y_u.astype(np.float32))
        # Initialize Fourier features
        # self.W = nn.Parameter(torch.randn(1, layers[0] // 2) * sigma, requires_grad=False)

        # Initialize network layers
        
        # Define the size of the Kernel
        self.D_u = X_u.shape[0]
        
        # Placeholder for NTK matrix
        self.K = None


        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=0.1,
                    scale_base=1.0,
                    scale_spline=1.0,
                    base_activation=torch.nn.SiLU,
                    grid_eps=0.02,
                    grid_range=grid_range,
                )
            )



    # Initialize network layers using Xavier initialization
    def initialize_NN(self, layers):
        layers_list = []
        num_layers = len(layers)
        for l in range(num_layers - 1):
            layers_list.append(nn.Linear(layers[l], layers[l + 1]))
            if l != num_layers - 2:
                layers_list.append(nn.Tanh())
        return nn.Sequential(*layers_list)
    
    # Neural network forward pass
    def forward(self, x: torch.Tensor, update_grid=False):
        for index, layer in enumerate(self.layers):
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
            if index < len(self.layers)-1: # 如果不是最后一层，拉到-1到1之间，最后一层不需要tanh
                x = torch.tanh(x)
        return x
    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )
    # Compute NTK
    def compute_NTK(self):
        G = compute_jacobian(self, self.X_u)
        K = result = np.dot(G, G.T)
        return K

    # Train the model
    def train_model(self, nIter, log_NTK=False, log_weights=False):
        self.loss_u_log = []
        self.train_error_log = []
        self.test_error_log = []
        self.K_log = []
        self.weights_log = []
        self.biases_log = []

        optimizer = optim.Adam(self.parameters(), lr = 0.001)
        criterion = nn.MSELoss()
        def closure():
            optimizer.zero_grad()
            u_pred = self.forward(self.X_u_tensor)
            loss = criterion(u_pred, self.Y_u_tensor)
            loss.backward()
            return loss
        start_time = time.time()
        for it in range(nIter):
            loss_value = optimizer.step(closure)
            self.loss_u_log.append(loss_value.data)
            if it % 10 == 0:
                elapsed = time.time() - start_time
                print('It: %d, Loss: %.3e, Time: %.2f' %
                      (it, loss_value.item(), elapsed))

                start_time = time.time()

            # Store the NTK matrix for every 100 iterations
            if log_NTK and it % 100 == 0:
                print("Compute NTK...")
                self.K = self.compute_NTK()
                self.K_log.append(self.K)

            # Store the weights and biases of the network for every 100 iterations
            if log_weights and it % 100 == 0:
                print("Weights stored...")
                self.weights_log.append([p.data.clone() for p in self.parameters()])
                self.biases_log.append([p.data.clone() for p in self.parameters() if p.requires_grad])

    # Evaluate predictions at test points
    def predict_u(self, X_star):
        with torch.no_grad():
            u_star = self.forward(torch.tensor(X_star, dtype=torch.float32))
        u_pred_numpy = u_star.numpy().astype(np.float64)
        return u_pred_numpy
    
class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output


    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features


    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        L1 and the entropy loss is for the feature selection, i.e., let the weight of the activation function be small.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )