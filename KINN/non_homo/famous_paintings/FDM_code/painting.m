%Number of grid points on [0,1]^2 
%i.e. uniform mesh with step h=1/(s-1)
s = 256;

%Create mesh (only needed for plotting)
[X,Y] = meshgrid(0:(1/(s-1)):1);

% Load images from .mat files
picasuo_data = load('../painting/picasuo_bw.mat');
salvator_data = load('../painting/salvator_bw.mat');
sky_data = load('../painting/sky_bw.mat');

% Extract the arrays from the loaded data
picasuo_bw = picasuo_data.picasuo_bw;
salvator_bw = salvator_data.salvator_bw;
sky_bw = sky_data.sky_bw;

% Choose one of the images to use as norm_a
norm_a = double(salvator_bw); % or salvator_bw or sky_bw


%Forcing function, f(x) = 1 
f = ones(s,s);

%Solve PDE: - div(a(x)*grad(p(x))) = f(x)
T = solve(norm_a,f);
save('./picasuo.mat', 'T');

%Plot coefficients and solutions
subplot(1,2,1)
surf(X,Y,norm_a); 
view(2); 
shading interp;
colorbar;
subplot(1,2,2)
surf(X,Y,T); 
view(2); 
shading interp;
colorbar;
