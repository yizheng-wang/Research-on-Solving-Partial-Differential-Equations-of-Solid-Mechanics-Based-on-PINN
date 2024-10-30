from dem_hyperelasticity.importlib import *
from dem_hyperelasticity import Utility as util
from dem_hyperelasticity.IntegrationLoss import *
from dem_hyperelasticity.MultiLayerNet import *
from dem_hyperelasticity.EnergyModel import *


class DeepEnergyMethod:
    # Instance attributes
    def __init__(self, model, numIntType, energy, dim):
        # self.data = data
        self.model = MultiLayerNet(model[0], model[1], model[2])
        self.model = self.model.to(dev)
        self.intLoss = IntegrationLoss(numIntType, dim)
        self.energy = EnergyModel(energy, dim)

    def train_model(self, shape, dxdydz, data, neumannBC, dirichletBC, LHD, iteration, type_energy):
        x = torch.from_numpy(data).float()
        x = x.to(dev)
        x.requires_grad_(True)
        # get tensor inputs and outputs for boundary conditions
        # -------------------------------------------------------------------------------
        #                             Dirichlet BC
        # -------------------------------------------------------------------------------
        dirBC_coordinates = {}  # declare a dictionary
        dirBC_values = {}  # declare a dictionary
        dirBC_penalty = {}
        for i, keyi in enumerate(dirichletBC):
            dirBC_coordinates[i] = torch.from_numpy(dirichletBC[keyi]['coord']).float().to(dev)
            dirBC_values[i] = torch.from_numpy(dirichletBC[keyi]['known_value']).float().to(dev)
            dirBC_penalty[i] = torch.tensor(dirichletBC[keyi]['penalty']).float().to(dev)
        # -------------------------------------------------------------------------------
        #                           Neumann BC
        # -------------------------------------------------------------------------------
        neuBC_coordinates = {}  # declare a dictionary
        neuBC_values = {}  # declare a dictionary
        neuBC_penalty = {}
        for i, keyi in enumerate(neumannBC):
            neuBC_coordinates[i] = torch.from_numpy(neumannBC[keyi]['coord']).float().to(dev)
            neuBC_coordinates[i].requires_grad_(True)
            neuBC_values[i] = torch.from_numpy(neumannBC[keyi]['known_value']).float().to(dev)
            neuBC_penalty[i] = torch.tensor(neumannBC[keyi]['penalty']).float().to(dev)

        # ----------------------------------------------------------------------------------
        # Minimizing loss function (energy and boundary conditions)
        # ----------------------------------------------------------------------------------
        optimizer = torch.optim.LBFGS(self.model.parameters(), lr=0.5, max_iter=20)
        start_time = time.time()
        energy_loss_array = []
        boundary_loss_array = []
        loss_array = []
        for t in range(iteration):
            # Zero gradients, perform a backward pass, and update the weights.
            def closure():
                it_time = time.time()
                # ----------------------------------------------------------------------------------
                # Internal Energy
                # ----------------------------------------------------------------------------------
                u_pred = self.model(x)  # prediction of primary variables
                u_pred.double()
                # Strain energy equations = Internal Energy
                storedEnergy = self.energy.getStoredEnergy(u_pred, x)
                dim = len(dxdydz)
                volume = LHD[0] * LHD[1] * LHD[2]

                dom_crit = volume * self.loss_sum(storedEnergy)
                if dim == 2:
                    internal2 = self.intLoss.approxIntegration(storedEnergy, x, shape=shape)
                elif dim == 3:
                    internal2 = self.trapz3D(storedEnergy, dx=dxdydz[0], dy=dxdydz[1], dz=dxdydz[2], shape=shape)
                # ----------------------------------------------------------------------------------
                # External Energy
                # ----------------------------------------------------------------------------------
                bc_n_crit = torch.zeros(len(neuBC_coordinates))
                external = torch.zeros(len(neuBC_coordinates))
                external2 = torch.zeros(len(neuBC_coordinates))
                for i, vali in enumerate(neuBC_coordinates):
                    if i == 0:
                        neu_u_pred = self.model(neuBC_coordinates[i])
                        area = LHD[1] * LHD[2]
                        fext = torch.bmm((neu_u_pred + neuBC_coordinates[i]).unsqueeze(1), neuBC_values[i].unsqueeze(2))
                        bc_n_crit[i] = area * self.loss_sum(fext) * neuBC_penalty[i]
                        if dim == 2:
                            external[i] = self.trapz1D(fext, neuBC_coordinates[i][:, 1])
                            external2[i] = self.trapz1D(fext, dx=dxdydz[1])
                        elif dim == 3:
                            external2[i] = self.trapz2D(fext, dx=dxdydz[1], dy=dxdydz[2], shape=[shape[1], shape[2]])
                    else:
                        print("Not yet implemented !!! Please contact the author to ask !!!")
                        exit()
                # ----------------------------------------------------------------------------------
                # Dirichlet boundary conditions
                # ----------------------------------------------------------------------------------
                # boundary 1 x - direction
                bc_u_crit = torch.zeros((len(dirBC_coordinates)))
                for i, vali in enumerate(dirBC_coordinates):
                    dir_u_pred = self.model(dirBC_coordinates[i])
                    bc_u_crit[i] = self.loss_squared_sum(dir_u_pred, dirBC_values[i]) * dirBC_penalty[i]
                # ----------------------------------------------------------------------------------
                # Compute and print loss
                # ----------------------------------------------------------------------------------
                energy_loss = internal2 - torch.sum(external2)
                boundary_loss = torch.sum(bc_u_crit)
                loss = energy_loss + boundary_loss
                optimizer.zero_grad()
                loss.backward()
                print('Iter: %d Loss: %.9e Energy: %.9e Boundary: %.9e Time: %.3e'
                      % (t + 1, loss.item(), energy_loss.item(), boundary_loss.item(), time.time() - it_time))
                energy_loss_array.append(energy_loss.data)
                boundary_loss_array.append(boundary_loss.data)
                loss_array.append(loss.data)
                return loss
            optimizer.step(closure)
        elapsed = time.time() - start_time
        print('Training time: %.4f' % elapsed)

    def __trapz(self, y, x=None, dx=1.0, axis=-1):
        # y = np.asanyarray(y)
        if x is None:
            d = dx
        else:
            d = x[1:] - x[0:-1]
            # reshape to correct shape
            shape = [1] * y.ndimension()
            shape[axis] = d.shape[0]
            d = d.reshape(shape)
        nd = y.ndimension()
        slice1 = [slice(None)] * nd
        slice2 = [slice(None)] * nd
        slice1[axis] = slice(1, None)
        slice2[axis] = slice(None, -1)
        ret = torch.sum(d * (y[tuple(slice1)] + y[tuple(slice2)]) / 2.0, axis)
        return ret

    def trapz1D(self, y, x=None, dx=1.0, axis=-1):
        y1D = y.flatten()
        if x is not None:
            x1D = x.flatten()
            return self.__trapz(y1D, x1D, dx=dx, axis=axis)
        else:
            return self.__trapz(y1D, dx=dx)

    def trapz2D(self, f, xy=None, dx=None, dy=None, shape=None):
        f2D = f.reshape(shape[0], shape[1])
        if dx is None and dy is None:
            x = xy[:, 0].flatten().reshape(shape[0], shape[1])
            y = xy[:, 1].flatten().reshape(shape[0], shape[1])
            return self.__trapz(self.__trapz(f2D, y[0, :]), x[:, 0])
        else:
            return self.__trapz(self.__trapz(f2D, dx=dy), dx=dx)

    def trapz3D(self, f, xyz=None, dx=None, dy=None, dz=None, shape=None):
        f3D = f.reshape(shape[0], shape[1], shape[2])
        if dx is None and dy is None and dz is None:
            print("dxdydz - trapz3D - Need to implement !!!")
        else:
            return self.__trapz(self.__trapz(self.__trapz(f3D, dx=dz), dx=dy), dx=dx)

    # --------------------------------------------------------------------------------
    # Purpose: After training model, predict solutions with some data input
    # --------------------------------------------------------------------------------
    def evaluate_model(self, x_space, y_space, z_space):
        surfaceUx = np.zeros([len(y_space), len(x_space), len(z_space)])
        surfaceUy = np.zeros([len(y_space), len(x_space), len(z_space)])
        surfaceUz = np.zeros([len(y_space), len(x_space), len(z_space)])
        for i, y in enumerate(y_space):
            for j, x in enumerate(x_space):
                for k, z in enumerate(z_space):
                    t_tensor = torch.tensor([x, y, z]).unsqueeze(0)
                    tRes = self.model(t_tensor).detach().cpu().numpy()[0]
                    surfaceUx[i][j][k] = tRes[0]
                    surfaceUy[i][j][k] = tRes[1]
                    surfaceUz[i][j][k] = tRes[2]
        return surfaceUx, surfaceUy, surfaceUz

    # --------------------------------------------------------------------------------
    # Purpose: After training model, predict solutions with some data input in 2D
    # --------------------------------------------------------------------------------
    def evaluate_model2d(self, x_space, y_space):
        z_space = np.array([0])
        surfaceUx = np.zeros([len(y_space), len(x_space), len(z_space)])
        surfaceUy = np.zeros([len(y_space), len(x_space), len(z_space)])
        surfaceUz = np.zeros([len(y_space), len(x_space), len(z_space)])
        for i, y in enumerate(y_space):
            for j, x in enumerate(x_space):
                for k, z in enumerate(z_space):
                    t_tensor = torch.tensor([x, y]).unsqueeze(0)
                    tRes = self.model(t_tensor).detach().cpu().numpy()[0]
                    surfaceUx[i][j][k] = tRes[0]
                    surfaceUy[i][j][k] = tRes[1]
                    surfaceUz[i][j][k] = 0
        return surfaceUx, surfaceUy, surfaceUz

    # --------------------------------------------------------------------------------
    # Purpose: Evaluate data
    # --------------------------------------------------------------------------------
    def evaluate_data(self, data):
        new_position = np.zeros(np.shape(data))
        disp = np.zeros(np.shape(data))
        for i, vali in enumerate(data):
            t_tensor = torch.tensor([vali[0], vali[1]]).unsqueeze(0)
            tRes = self.model(t_tensor).detach().cpu().numpy()[0]
            disp[i, :] = np.copy(tRes)
            new_position[i, :] = vali + tRes
        return new_position, disp

    # --------------------------------------------------------------------------------
    # method: loss sum for the energy part
    # --------------------------------------------------------------------------------
    @staticmethod
    def loss_sum(tinput):
        return torch.sum(tinput) / tinput.data.nelement()

    # --------------------------------------------------------------------------------
    # purpose: loss square sum for the boundary part
    # --------------------------------------------------------------------------------
    @staticmethod
    def loss_squared_sum(tinput, target):
        row, column = tinput.shape
        loss = 0
        for j in range(column):
            loss += torch.sum((tinput[:, j] - target[:, j]) ** 2) / tinput[:, j].data.nelement()
        return loss


if __name__ == '__main__':
    from Hyperelasticity import data_testing
    data = data_testing.get_data()
