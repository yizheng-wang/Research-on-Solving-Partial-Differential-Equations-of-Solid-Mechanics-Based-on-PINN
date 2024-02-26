function [CH,X0,X] = homo3D_Alloutput(lx,ly,lz,lambda,mu,voxel)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% lx       = Unit cell length in x-direction.
% ly       = Unit cell length in y-direction.
% lz       = Unit cell length in z-direction.
% lambda   = Lame's first parameter for solid materials.
% mu       = Lame's second parameter for solid materials.
% voxel    = Material indicator matrix. Used to determine nelx/nely/nelz
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% INITIALIZE
[nelx, nely, nelz] = size(voxel); %size of voxel model along x,y and z axis
% Stiffness matrix
dx = lx/nelx; dy = ly/nely; dz = lz/nelz;
nel = nelx*nely*nelz;
[keLambda, keMu, feLambda, feMu] = hexahedron(dx/2,dy/2,dz/2);
% Node numbers and element degrees of freedom for full (not periodic) mesh
nodenrs = reshape(1:(1+nelx)*(1+nely)*(1+nelz),1+nelx,1+nely,1+nelz);
edofVec = reshape(3*nodenrs(1:end-1,1:end-1,1:end-1)+1,nel,1);
addx = [0 1 2 3*nelx+[3 4 5 0 1 2] -3 -2 -1];
addxy = 3*(nely+1)*(nelx+1)+addx;
edof = repmat(edofVec,1,24) + repmat([addx addxy],nel,1);
%% IMPOOSE PERIODIC BOUNDARY CONDITIONS
% Use original edofMat to index into list with the periodic dofs
nn = (nelx+1)*(nely+1)*(nelz+1); % Total number of nodes
nnP = (nelx)*(nely)*(nelz);    % Total number of unique nodes
nnPArray = reshape(1:nnP, nelx, nely, nelz);
% Extend with a mirror of the back border
nnPArray(end+1,:,:) = nnPArray(1,:,:);
% Extend with a mirror of the left border
nnPArray(:, end+1, :) = nnPArray(:,1,:);
% Extend with a mirror of the top border
nnPArray(:, :, end+1) = nnPArray(:,:,1);
% Make a vector into which we can index using edofMat:
dofVector = zeros(3*nn, 1);
dofVector(1:3:end) = 3*nnPArray(:)-2;
dofVector(2:3:end) = 3*nnPArray(:)-1;
dofVector(3:3:end) = 3*nnPArray(:);
edof = dofVector(edof);
ndof = 3*nnP;
%% ASSEMBLE GLOBAL STIFFNESS MATRIX AND LOAD VECTORS
% Indexing vectors
iK = kron(edof,ones(24,1))';
jK = kron(edof,ones(1,24))';
% Material properties assigned to voxels with materials
% lambda0 = lambda;
% mu0 = mu;

% lambda = lambda*(voxel==1);  mu = mu*(voxel==1);
% factor_voxel = max(max(max(voxel)));
% lambda = lambda*voxel/factor_voxel;  mu = mu*voxel/factor_voxel;
lambda = lambda*voxel; mu = mu*voxel;

% lambda(lambda==0) = lambda0*1e-6;
% mu(mu==0) = mu0*1e-6;
% The corresponding stiffness matrix entries
sK = keLambda(:)*lambda(:).' + keMu(:)*mu(:).';
K = sparse(iK(:), jK(:), sK(:), ndof, ndof);
K = 1/2*(K+K');
% Assembly three load cases corresponding to the three strain cases
iF = repmat(edof',6,1);
jF = [ones(24,nel); 2*ones(24,nel); 3*ones(24,nel);...
    4*ones(24,nel); 5*ones(24,nel); 6*ones(24,nel);];
sF = feLambda(:)*lambda(:).'+feMu(:)*mu(:).';
F  = sparse(iF(:), jF(:), sF(:), ndof, 6);
%% SOLUTION
% solve by PCG method, remember to constrain one node
activedofs = edof(voxel>0,:); activedofs = sort(unique(activedofs(:)));
% activedofs = edof; activedofs = sort(unique(activedofs(:)));
X = zeros(ndof,6);
alpha = 0.1;
L = ichol(K(activedofs(4:end),activedofs(4:end)),struct('type','ict','droptol',1e-3,'diagcomp',alpha));
for i = 1:6
    X(activedofs(4:end),i) = pcg(K(activedofs(4:end),...
        activedofs(4:end)),F(activedofs(4:end),i),1e-10,300,L,L');
end
% X(activedofs(4:end),:) = K(activedofs(4:end),activedofs(4:end))...
%     \F(activedofs(4:end),:);    % Solving by direct method
%% HOMOGENIZATION
% The displacement vectors corresponding to the unit strain cases
X0 = zeros(nel, 24, 6);
% The element displacements for the six unit strains
X0_e = zeros(24, 6);
%fix degrees of nodes [1 2 3 5 6 12];
ke = keMu + keLambda; % Here the exact ratio does not matter, because
fe = feMu + feLambda; % it is reflected in the load vector
X0_e([4 7:11 13:24],:) = ke([4 7:11 13:24],[4 7:11 13:24])...
                           \fe([4 7:11 13:24],:);
X0(:,:,1) = kron(X0_e(:,1)', ones(nel,1)); % epsilon0_11 = (1,0,0,0,0,0)
X0(:,:,2) = kron(X0_e(:,2)', ones(nel,1)); % epsilon0_22 = (0,1,0,0,0,0)
X0(:,:,3) = kron(X0_e(:,3)', ones(nel,1)); % epsilon0_33 = (0,0,1,0,0,0)
X0(:,:,4) = kron(X0_e(:,4)', ones(nel,1)); % epsilon0_12 = (0,0,0,1,0,0)
X0(:,:,5) = kron(X0_e(:,5)', ones(nel,1)); % epsilon0_23 = (0,0,0,0,1,0)
X0(:,:,6) = kron(X0_e(:,6)', ones(nel,1)); % epsilon0_13 = (0,0,0,0,0,1)
CH = zeros(6);
volume = lx*ly*lz;
for i = 1:6
    for j = 1:6
        sum_L = ((X0(:,:,i) - X(edof+(i-1)*ndof))*keLambda).*...
            (X0(:,:,j) - X(edof+(j-1)*ndof));
        sum_M = ((X0(:,:,i) - X(edof+(i-1)*ndof))*keMu).*...
            (X0(:,:,j) - X(edof+(j-1)*ndof));
        sum_L = reshape(sum(sum_L,2), nelx, nely, nelz);
        sum_M = reshape(sum(sum_M,2), nelx, nely, nelz);
        % Homogenized elasticity tensor
        CH(i,j) = 1/volume*sum(sum(sum(lambda.*sum_L + mu.*sum_M)));
    end
end
end
%% COMPUTE ELEMENT STIFFNESS MATRIX AND LOAD VECTOR
function [keLambda, keMu, feLambda, feMu] = hexahedron(a, b, c)
% Constitutive matrix contributions
CMu = diag([2 2 2 1 1 1]); CLambda = zeros(6); CLambda(1:3,1:3) = 1;
% Three Gauss points in both directions
xx = [-sqrt(3/5), 0, sqrt(3/5)]; yy = xx; zz = xx;
ww = [5/9, 8/9, 5/9];
% Initialize
keLambda = zeros(24,24); keMu = zeros(24,24);
feLambda = zeros(24,6); feMu = zeros(24,6);
for ii = 1:length(xx)
    for jj = 1:length(yy)
        for kk = 1:length(zz)
            %integration point
            x = xx(ii); y = yy(jj); z = zz(kk);
            %stress strain displacement matrix
            qx = [ -((y-1)*(z-1))/8, ((y-1)*(z-1))/8, -((y+1)*(z-1))/8,...
                ((y+1)*(z-1))/8, ((y-1)*(z+1))/8, -((y-1)*(z+1))/8,...
                ((y+1)*(z+1))/8, -((y+1)*(z+1))/8];
            qy = [ -((x-1)*(z-1))/8, ((x+1)*(z-1))/8, -((x+1)*(z-1))/8,...
                ((x-1)*(z-1))/8, ((x-1)*(z+1))/8, -((x+1)*(z+1))/8,...
                ((x+1)*(z+1))/8, -((x-1)*(z+1))/8];
            qz = [ -((x-1)*(y-1))/8, ((x+1)*(y-1))/8, -((x+1)*(y+1))/8,...
                ((x-1)*(y+1))/8, ((x-1)*(y-1))/8, -((x+1)*(y-1))/8,...
                ((x+1)*(y+1))/8, -((x-1)*(y+1))/8];
            % Jacobian
            J = [qx; qy; qz]*[-a a a -a -a a a -a; -b -b b b -b -b b b;...
                -c -c -c -c c c c c]';
            qxyz = J\[qx;qy;qz];
            B_e = zeros(6,3,8);
            for i_B = 1:8
                B_e(:,:,i_B) = [qxyz(1,i_B)   0             0;
                                0             qxyz(2,i_B)   0;
                                0             0             qxyz(3,i_B);
                                qxyz(2,i_B)   qxyz(1,i_B)   0;
                                0             qxyz(3,i_B)   qxyz(2,i_B);
                                qxyz(3,i_B)   0             qxyz(1,i_B)];
            end
            B = [B_e(:,:,1) B_e(:,:,2) B_e(:,:,3) B_e(:,:,4) B_e(:,:,5)...
                B_e(:,:,6) B_e(:,:,7) B_e(:,:,8)];
            % Weight factor at this point
            weight = det(J)*ww(ii) * ww(jj) * ww(kk);
            % Element matrices
            keLambda = keLambda + weight * B' * CLambda * B;
            keMu = keMu + weight * B' * CMu * B;
            % Element loads
            feLambda = feLambda + weight * B' * CLambda;       
            feMu = feMu + weight * B' * CMu; 
        end
    end
end
end