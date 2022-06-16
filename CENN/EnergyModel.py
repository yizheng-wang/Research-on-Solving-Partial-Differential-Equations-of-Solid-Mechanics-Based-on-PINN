from config import *


class EnergyModel:
    # ---------------------------------------------------------------------------------------------------------------------------
    def __init__(self, energy, dim, E=None, nu=None, param_c1=None, param_c2=None, param_c=None, rou = 1000):
        self.type = energy
        self.dim = dim
        if self.type == 'neohookean':
            self.mu = E / (2 * (1 + nu))
            self.lam = (E * nu) / ((1 + nu) * (1 - 2 * nu))
        if self.type == 'mooneyrivlin':
            self.param_c1 = param_c1
            self.param_c2 = param_c2
            self.param_c = param_c
            self.param_d = 2 * (self.param_c1 + 2 * self.param_c2)
        if self.type == 'elasticityHW':
            if dim == 2:
                self.D11_mat = E/(1-nu**2)
                self.D22_mat = E/(1-nu**2)
                self.D12_mat = E*nu/(1-nu**2)
                self.D21_mat = E*nu/(1-nu**2)
                self.D33_mat = E/(2*(1+nu))
            if dim ==3: # 3D problems defined in the future
                pass
        if self.type == 'elasticityMP':
            if dim == 2:
                self.D11_mat = E/(1-nu**2)
                self.D22_mat = E/(1-nu**2)
                self.D12_mat = E*nu/(1-nu**2)
                self.D21_mat = E*nu/(1-nu**2)
                self.D33_mat = E/(2*(1+nu))
            if dim ==3: # 3D problems defined in the future
                pass
        if self.type == 'elasticityMCP':
            if dim == 2:
                self.E = E
                self.nu = nu
                self.mu = E / (2 * (1 + nu))
            if dim ==3: # 3D problems defined in the future
                pass
        if self.type == 'elasticityHam':
            if dim == 2:
                self.D11_mat = E/(1-nu**2)
                self.D22_mat = E/(1-nu**2)
                self.D12_mat = E*nu/(1-nu**2)
                self.D21_mat = E*nu/(1-nu**2)
                self.D33_mat = E/(2*(1+nu))
                self.rou = rou
            if dim ==3: # 3D problems defined in the future
                pass
    def getStoredEnergy(self, u, x): # get the energy density from the displacement and the coordinate
        if self.type == 'neohookean':
            if self.dim == 2:
                return self.NeoHookean2D(u, x)
            if self.dim == 3:
                return self.NeoHookean3D(u, x) 
            
        if self.type == 'mooneyrivlin':
            if self.dim == 2:
                return self.MooneyRivlin2D(u, x) 
            if self.dim == 3:
                return self.MooneyRivlin3D(u, x)
        if self.type == 'elasticityHW':
            if self.dim == 2:
                return self.Elasticity2DHW(u, x)
            if self.dim == 3:
                return self.Elasticity3DHW(u, x)
        if self.type == 'elasticityMP': # principle of minimum potential energy
            if self.dim == 2:
                return self.Elasticity2DMP(u, x)
            if self.dim == 3:
                return self.Elasticity3DMP(u, x)
        if self.type == 'elasticityMCP': # principle of minimum complementary energy
            if self.dim == 2:
                return self.Elasticity2DMCP(u, x) # u is Airy function
            if self.dim == 3:
                return self.Elasticity3DMCP(u, x)
        if self.type == 'elasticityHam': # Hamilton in the future
            if self.dim == 2:
                return self.Elasticity2DMP(u, x)
            if self.dim == 3:
                return self.Elasticity3DMP(u, x)
        if self.type == 'crack_third': # Laplacian equation (3 type of crack)
            return self.crack_third(u, x)
    def getkineticEnergy(self, ust, x):
        duxdxy = grad(ust[:, 0].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        duydxy = grad(ust[:, 1].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]        
        dudt = duxdxy[:, 2].unsqueeze(1) # v in the x direction
        dvdt = duydxy[:, 2].unsqueeze(1) # v in the y direction
        # get the kinematic density
        kineticDensity = 0.5 * self.rou * (dudt**2 + dvdt**2) 
        return kineticDensity
        
    def getEssentialEnergy(self, ust, uv_pre, normal):
        '''
        

        Parameters
        ----------
        ust : tensor 2 orders
            input dimensional is displacement, strain and stress .
        u_pre : tensor 2 orders 
            x displacement and y displacement.
        normal : normal
            x and y normal direction of dir condition.

        Returns
        -------
        Essential Energy in functional of hw.

        '''
        #Input is physically meaningful and analyzed column by column
        if self.dim == 2:#Checked the following, it's right
            u_pred = ust[:, 0].unsqueeze(1)
            v_pred = ust[:, 1].unsqueeze(1)
            u_pre = uv_pre[:, 0].unsqueeze(1)
            v_pre = uv_pre[:, 1].unsqueeze(1)
            t11 = ust[:, 5].unsqueeze(1)
            t22 = ust[:, 6].unsqueeze(1)
            t12 = ust[:, 7].unsqueeze(1)
            normx = normal[:, 0].unsqueeze(1)
            normy = normal[:, 1].unsqueeze(1)
            #Obtain the difference between the predicted displacement and the given displacement
            u_loss = u_pred - u_pre
            v_loss = v_pred - v_pre
            stress_vectorx = t11 * normx + t12 * normy
            stress_vectory = t12 * normx + t22 * normy
            #Output the dot product of the stress vector and the predicted displacement and the given displacement
            EssentialEnergy = u_loss * stress_vectorx + v_loss * stress_vectory
            return EssentialEnergy
        if self.dim == 3 :#3D will be defined later
            pass

    def MooneyRivlin3D(self, u, x):
        duxdxyz = grad(u[:, 0].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        duydxyz = grad(u[:, 1].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        duzdxyz = grad(u[:, 2].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        Fxx = duxdxyz[:, 0].unsqueeze(1) + 1
        Fxy = duxdxyz[:, 1].unsqueeze(1) + 0
        Fxz = duxdxyz[:, 2].unsqueeze(1) + 0
        Fyx = duydxyz[:, 0].unsqueeze(1) + 0
        Fyy = duydxyz[:, 1].unsqueeze(1) + 1
        Fyz = duydxyz[:, 2].unsqueeze(1) + 0
        Fzx = duzdxyz[:, 0].unsqueeze(1) + 0
        Fzy = duzdxyz[:, 1].unsqueeze(1) + 0
        Fzz = duzdxyz[:, 2].unsqueeze(1) + 1
        detF = Fxx * (Fyy * Fzz - Fyz * Fzy) - Fxy * (Fyx * Fzz - Fyz * Fzx) + Fxz * (Fyx * Fzy - Fyy * Fzx)
        C11 = Fxx ** 2 + Fyx ** 2 + Fzx ** 2
        C12 = Fxx * Fxy + Fyx * Fyy + Fzx * Fzy
        C13 = Fxx * Fxz + Fyx * Fyz + Fzx * Fzz
        C21 = Fxy * Fxx + Fyy * Fyx + Fzy * Fzx
        C22 = Fxy ** 2 + Fyy ** 2 + Fzy ** 2
        C23 = Fxy * Fxz + Fyy * Fyz + Fzy * Fzz
        C31 = Fxz * Fxx + Fyz * Fyx + Fzz * Fzx
        C32 = Fxz * Fxy + Fyz * Fyy + Fzz * Fzy
        C33 = Fxz ** 2 + Fyz ** 2 + Fzz ** 2
        trC = C11 + C22 + C33
        trC2 = C11*C11 + C12*C21 + C13*C31 + C21*C12 + C22*C22 + C23*C32 + C31*C13 + C32*C23 + C33*C33
        I1 = trC
        I2 = 0.5 * (trC*trC - trC2)
        J = detF
        strainEnergy = self.param_c * (J - 1) ** 2 - self.param_d * torch.log(J) + self.param_c1 * (
                I1 - 3) + self.param_c2 * (I2 - 3)
        return strainEnergy


    def MooneyRivlin2D(self, u, x):
        duxdxy = grad(u[:, 0].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        duydxy = grad(u[:, 1].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        Fxx = duxdxy[:, 0].unsqueeze(1) + 1
        Fxy = duxdxy[:, 1].unsqueeze(1) + 0
        Fyx = duydxy[:, 0].unsqueeze(1) + 0
        Fyy = duydxy[:, 1].unsqueeze(1) + 1
        detF = Fxx * Fyy - Fxy * Fyx
        C11 = Fxx * Fxx + Fyx * Fyx
        C12 = Fxx * Fxy + Fyx * Fyy
        C21 = Fxy * Fxx + Fyy * Fyx
        C22 = Fxy * Fxy + Fyy * Fyy
        J = detF
        traceC = C11 + C22
        I1 = traceC
        trace_C2 = C11 * C11 + C12 * C21 + C21 * C12 + C22 * C22
        I2 = 0.5 * (traceC ** 2 - trace_C2)
        strainEnergy = self.param_c * (J - 1) ** 2 - self.param_d * torch.log(J) + self.param_c1 * (I1 - 2) + self.param_c2 * (I2 - 1)
        return strainEnergy

    # ---------------------------------------------------------------------------------------
    # Purpose: calculate Neo-Hookean potential energy in 3D
    # ---------------------------------------------------------------------------------------
    def NeoHookean3D(self, u, x):
        duxdxyz = grad(u[:, 0].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        duydxyz = grad(u[:, 1].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        duzdxyz = grad(u[:, 2].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        Fxx = duxdxyz[:, 0]. Unsqueeze (1) + 1 #will shift gradient into F
        Fxy = duxdxyz[:, 1].unsqueeze(1) + 0
        Fxz = duxdxyz[:, 2].unsqueeze(1) + 0
        Fyx = duydxyz[:, 0].unsqueeze(1) + 0
        Fyy = duydxyz[:, 1].unsqueeze(1) + 1
        Fyz = duydxyz[:, 2].unsqueeze(1) + 0
        Fzx = duzdxyz[:, 0].unsqueeze(1) + 0
        Fzy = duzdxyz[:, 1].unsqueeze(1) + 0
        Fzz = duzdxyz[:, 2].unsqueeze(1) + 1
        detF = Fxx * (Fyy * Fzz - Fyz * Fzy) - Fxy * (Fyx * Fzz - Fyz * Fzx) + Fxz * (Fyx * Fzy -Fyy * Fzx) #self-written determinant
        trC = Fxx ** 2 + Fxy ** 2 + Fxz ** 2 + Fyx ** 2 + Fyy ** 2 + Fyz ** 2 + Fzx ** 2 + Fzy ** 2 + Fzz ** 2 #green tensor
        strainEnergy = 0.5 * self.lam * (torch.log (detF) * torch.log (detF)) -self.mu * torch.log (detF) + 0.5 * self.mu * (trC - 3) #Defines the energy density of the neocook
        return strainEnergy #Returns a two-dimensional column vector containing the strain energy densities of all points

    # ---------------------------------------------------------------------------------------
    # Purpose: calculate Neo-Hookean potential energy in 2D
    # ---------------------------------------------------------------------------------------
    def NeoHookean2D(self, u, x):
        duxdxy = grad(u[:, 0].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        duydxy = grad(u[:, 1].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        Fxx = duxdxy[:, 0].unsqueeze(1) + 1
        Fxy = duxdxy[:, 1].unsqueeze(1) + 0
        Fyx = duydxy[:, 0].unsqueeze(1) + 0
        Fyy = duydxy[:, 1].unsqueeze(1) + 1
        detF = Fxx * Fyy - Fxy * Fyx
        trC = Fxx ** 2 + Fxy ** 2 + Fyx ** 2 + Fyy ** 2
        strainEnergy = 0.5 * self.lam * (torch.log(detF) * torch.log(detF)) - self.mu * torch.log(detF) + 0.5 * self.mu * (trC - 2)
        return strainEnergy
    def Elasticity2DHW(self, ust, x):
        duxdxy = grad(ust[:, 0].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        duydxy = grad(ust[:, 1].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]        
        dudx = duxdxy[:, 0].unsqueeze(1)
        dudy = duxdxy[:, 1].unsqueeze(1)
        dvdx = duydxy[:, 0].unsqueeze(1)
        dvdy = duydxy[:, 1].unsqueeze(1)
        sxx = ust[:, 2].unsqueeze(1)
        syy = ust[:, 3].unsqueeze(1)
        s2xy = ust[:, 4].unsqueeze(1)
        txx = ust[:, 5].unsqueeze(1)
        tyy = ust[:, 6].unsqueeze(1)
        txy = ust[:, 7].unsqueeze(1)
        Sg11 = sxx - dudx #define nominal strain minus the geometric strain
        sg22 = syy - dvdy
        sg12 = s2xy - (dudy + dvdx) #where sg12 is the nominal strain minus twice the geometric equation, is vigot said
        geoEnergy = txx * sg11 + tyy * sg22 + txy * sg12 #obtain the functional of the geometry equation
        
        strainEnergy = 0.5 * (self.D11_mat * sxx ** 2  + 2*self.D12_mat * sxx * syy + self.D22_mat * syy ** 2 + self.D33_mat * s2xy ** 2)
        #Obtain functional of strain energy density
        return strainEnergy, geoEnergy


    def Elasticity3DHW(self, u, x):
        duxdxy = grad(u[:, 0].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        duydxy = grad(u[:, 1].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        Return #later in the definition of three-dimensional elasticity problem to write this function
    
    def Elasticity2DMP(self, ust, x):
        duxdxy = grad(ust[:, 0].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        duydxy = grad(ust[:, 1].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]        
        dudx = duxdxy[:, 0].unsqueeze(1)
        dudy = duxdxy[:, 1].unsqueeze(1)
        dvdx = duydxy[:, 0].unsqueeze(1)
        dvdy = duydxy[:, 1].unsqueeze(1)
        sxx = dudx
        syy = dvdy
        s2xy = dudy + dvdx 


        #Tensor analysis theory analysis, this is right
        strainEnergy = 0.5 * (self.D11_mat * sxx ** 2  + 2*self.D12_mat * sxx * syy + self.D22_mat * syy ** 2 + self.D33_mat * s2xy ** 2)
        #Obtain functional of strain energy density
        return strainEnergy
        
        
        
    def Elasticity3DMP(self, u, x):
        duxdxy = grad(u[:, 0].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        duydxy = grad(u[:, 1].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        Return #later in the definition of three-dimensional elasticity problem to write this function
    
    def Elasticity2DMCP(self, fai, x):
        dfdxy = grad(fai[:, 0].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        dfxdxy = grad(dfdxy[:, 0].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]        
        dfydxy = grad(dfdxy[:, 1].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]    
        sxx = dfydxy[:, 1].unsqueeze(1)
        syy = dfxdxy[:, 0].unsqueeze(1)
        sxy = -dfxdxy[:, 1].unsqueeze(1)
        
        stressEnergy = 1/(4 * self.mu) * (sxx ** 2 + 2 * sxy ** 2 + syy ** 2) - self.nu / (2 * self.E) * (sxx + syy) ** 2
        #Obtain functional of strain energy density
        return stressEnergy
        
        
        
    def Elasticity3DMCP(self, u, x):
        duxdxy = grad(u[:, 0].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        duydxy = grad(u[:, 1].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        Return #later in the definition of three-dimensional elasticity problem to write this function
    
    def crack_third(self, u, x):# x is a two-dimensional coordinates, and u is a one-dimensional coordinates
        dudxy = grad(u[:, 0].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0] 
        dudx = dudxy[:, 0].unsqueeze(1)
        dudy = dudxy[:, 1].unsqueeze(1)
        energy = 0.5*(dudx**2 + dudy**2)
        return energy
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        