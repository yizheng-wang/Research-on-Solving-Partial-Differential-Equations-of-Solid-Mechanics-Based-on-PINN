import torch


class IntegrationLoss:
    def __init__(self, numIntType, dim):
        print("Constructor: IntegrationLoss ", numIntType, " in ", dim, " dimension ")
        self.type = numIntType
        self.dim = dim

    def lossInternalEnergy(self, f, x=None, dx=1.0, dy=1.0, dz=1.0, shape=None):
        return self.approxIntegration(f, x, dx, dy, dz, shape)

    def lossExternalEnergy(self, f, x=None, dx=1.0, dy=1.0, dz=1.0, shape=None):
        if self.type == 'trapezoidal':
            # print("Trapezoidal rule")
            if self.dim == 2:
                if x is not None:
                    return self.trapz1D(f, x=x)
                else:
                    return self.trapz1D(f, dx=dx)
            if self.dim == 3:
                if x is not None:
                    return self.trapz2D(f, xy=x, shape=shape)
                else:
                    return self.trapz2D(f, dx=dx, dy=dy, shape=shape)
        if self.type == 'simpson':
            # print("Simpson rule")
            if self.dim == 2:
                if x is not None:
                    return self.simps1D(f, x=x)
                else:
                    return self.simps1D(f, dx=dx)
            if self.dim == 3:
                if x is not None:
                    return self.simps2D(f, xy=x, shape=shape)
                else:
                    return self.simps2D(f, dx=dx, dy=dy, shape=shape)

    def approxIntegration(self, f, x=None, dx=1.0, dy=1.0, dz=1.0, shape=None):
        if self.type == 'trapezoidal':
            # print("Trapezoidal rule")
            if self.dim == 1:
                if x is not None:
                    return self.trapz1D(f, x=x)
                else:
                    return self.trapz1D(f, dx=dx)
            if self.dim == 2:
                if x is not None:
                    return self.trapz2D(f, xy=x, shape=shape)
                else:
                    return self.trapz2D(f, dx=dx, dy=dy, shape=shape)
            if self.dim == 3:
                if x is not None:
                    return self.trapz3D(f, xyz=x, shape=shape)
                else:
                    return self.trapz3D(f, dx=dx, dy=dy, dz=dz, shape=shape)
        if self.type == 'simpson':
            # print("Simpson rule")
            if self.dim == 1:
                if x is not None:
                    return self.simps1D(f, x=x)
                else:
                    return self.simps1D(f, dx=dx)
            if self.dim == 2:
                if x is not None:
                    return self.simps2D(f, xy=x, shape=shape)
                else:
                    return self.simps2D(f, dx=dx, dy=dy, shape=shape)
            if self.dim == 3:
                if x is not None:
                    return self.simps3D(f, xyz=x, shape=shape)
                else:
                    return self.simps3D(f, dx=dx, dy=dy, dz=dz, shape=shape)

    def trapz1D(self, y, x=None, dx=1.0, axis=-1):
        y1D = y.flatten()
        if x is not None:
            x1D = x.flatten()
            return self.trapz(y1D, x1D, dx=dx, axis=axis)
        else:
            return self.trapz(y1D, dx=dx)

    def trapz2D(self, f, xy=None, dx=None, dy=None, shape=None):
        f2D = f.reshape(shape[0], shape[1])
        if dx is None and dy is None:
            x = xy[:, 0].flatten().reshape(shape[0], shape[1])
            y = xy[:, 1].flatten().reshape(shape[0], shape[1])
            return self.trapz(self.trapz(f2D, y[0, :]), x[:, 0])
        else:
            return self.trapz(self.trapz(f2D, dx=dy), dx=dx)

    def trapz3D(self, f, xyz=None, dx=None, dy=None, dz=None, shape=None):
        f3D = f.reshape(shape[0], shape[1], shape[2])
        if dx is None and dy is None and dz is None:
            print("dxdydz - trapz3D - Need to implement !!!")
        else:
            return self.trapz(self.trapz(self.trapz(f3D, dx=dz), dx=dy), dx=dx)

    def simps1D(self, f, x=None, dx=1.0, axis=-1):
        f1D = f.flatten()
        if x is not None:
            x1D = x.flatten()
            return self.simps(f1D, x1D, dx=dx, axis=axis)
        else:
            return self.simps(f1D, dx=dx, axis=axis)

    def simps2D(self, f, xy=None, dx=None, dy=None, shape=None):
        f2D = f.reshape(shape[0], shape[1])
        if dx is None and dy is None:
            x = xy[:, 0].flatten().reshape(shape[0], shape[1])
            y = xy[:, 1].flatten().reshape(shape[0], shape[1])
            return self.simps(self.simps(f2D, y[0, :]), x[:, 0])
        else:
            return self.simps(self.simps(f2D, dx=dy), dx=dx)

    def simps3D(self, f, xyz=None, dx=None, dy=None, dz=None, shape=None):
        f3D = f.reshape(shape[0], shape[1], shape[2])
        if dx is None and dy is None and dz is None:
            print("dxdydz - trapz3D - Need to implement !!!")
        else:
            return self.simps(self.simps(self.simps(f3D, dx=dz), dx=dy), dx=dx)

    def montecarlo1D(self, fx, l):
        return l * torch.sum(fx) / fx.data.nelement()

    def montecarlo2D(self, fxy, lx, ly):
        area = lx * ly
        return area * torch.sum(fxy) / fxy.data.nelement()

    def montecarlo3D(self, fxyz, lx, ly, lz):
        volume = lx * ly * lz
        return volume * torch.sum(fxyz) / fxyz.data.nelement()

    def simps(self, y, x=None, dx=1, axis=-1, even='avg'):
        # import scipy.integrate as sp
        # sp.simps()
        # y = torch.tensor(y)
        nd = len(y.shape)
        N = y.shape[axis]
        last_dx = dx
        first_dx = dx
        returnshape = 0
        if x is not None:
            # x = torch.tensor(x)
            if len(x.shape) == 1:
                shapex = [1] * nd
                shapex[axis] = x.shape[0]
                saveshape = x.shape
                returnshape = 1
                x = x.reshape(tuple(shapex))
            elif len(x.shape) != len(y.shape):
                raise ValueError("If given, shape of x must be 1-d or the "
                                 "same as y.")
            if x.shape[axis] != N:
                raise ValueError("If given, length of x along axis must be the "
                                 "same as y.")
        if N % 2 == 0:
            val = 0.0
            result = 0.0
            slice1 = (slice(None),) * nd
            slice2 = (slice(None),) * nd
            if even not in ['avg', 'last', 'first']:
                raise ValueError("Parameter 'even' must be "
                                 "'avg', 'last', or 'first'.")
            # Compute using Simpson's rule on first intervals
            if even in ['avg', 'first']:
                slice1 = self.tupleset(slice1, axis, -1)
                slice2 = self.tupleset(slice2, axis, -2)
                if x is not None:
                    last_dx = x[slice1] - x[slice2]
                val += 0.5 * last_dx * (y[slice1] + y[slice2])
                result = self._basic_simps(y, 0, N - 3, x, dx, axis)
            # Compute using Simpson's rule on last set of intervals
            if even in ['avg', 'last']:
                slice1 = self.tupleset(slice1, axis, 0)
                slice2 = self.tupleset(slice2, axis, 1)
                if x is not None:
                    first_dx = x[tuple(slice2)] - x[tuple(slice1)]
                val += 0.5 * first_dx * (y[slice2] + y[slice1])
                result += self._basic_simps(y, 1, N - 2, x, dx, axis)
            if even == 'avg':
                val /= 2.0
                result /= 2.0
            result = result + val
        else:
            result = self._basic_simps(y, 0, N - 2, x, dx, axis)
        if returnshape:
            x = x.reshape(saveshape)
        return result

    def tupleset(self, t, i, value):
        l = list(t)
        l[i] = value
        return tuple(l)

    def _basic_simps(self, y, start, stop, x, dx, axis):
        nd = len(y.shape)
        if start is None:
            start = 0
        step = 2
        slice_all = (slice(None),) * nd
        slice0 = self.tupleset(slice_all, axis, slice(start, stop, step))
        slice1 = self.tupleset(slice_all, axis, slice(start + 1, stop + 1, step))
        slice2 = self.tupleset(slice_all, axis, slice(start + 2, stop + 2, step))

        if x is None:  # Even spaced Simpson's rule.
            result = torch.sum(dx / 3.0 * (y[slice0] + 4 * y[slice1] + y[slice2]), axis)
        else:
            # Account for possibly different spacings.
            #    Simpson's rule changes a bit.
            # h = np.diff(x, axis=axis)
            h = self.torch_diff_axis_0(x, axis=axis)
            sl0 = self.tupleset(slice_all, axis, slice(start, stop, step))
            sl1 = self.tupleset(slice_all, axis, slice(start + 1, stop + 1, step))
            h0 = h[sl0]
            h1 = h[sl1]
            hsum = h0 + h1
            hprod = h0 * h1
            h0divh1 = h0 / h1
            tmp = hsum / 6.0 * (y[slice0] * (2 - 1.0 / h0divh1) +
                                y[slice1] * hsum * hsum / hprod +
                                y[slice2] * (2 - h0divh1))
            result = torch.sum(tmp, dim=axis)
        return result

    def torch_diff_axis_0(self, a, axis):
        if axis == 0:
            return a[1:, 0:1] - a[:-1, 0:1]
        elif axis == -1:
            return a[1:] - a[:-1]
        else:
            print("Not implemented yet !!! function: torch_diff_axis_0 error !!!")
            exit()

    def trapz(self, y, x=None, dx=1.0, axis=-1):
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

