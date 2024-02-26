function otr = transform(itr,tmx)

%
% FUNCTION
% otr = transform(itr,tmx)
%
% DESCRIPTION
% transform 3D-tensor (Euclidean or Cartesion tensor) of any order (>0) to another coordinate system
%
% PARAMETERS
% otr = output tensor, after transformation; has the same dimensions as the input tensor
% itr = input tensor, before transformation; should be a 3-element vector, a 3x3 matrix, or a 3x3x3x... multidimensional array, each dimension containing 3 elements
% tmx = transformation matrix, 3x3 matrix that contains the direction cosines between the old and the new coordinate system
%

ne = numel(itr);                % number of tensor elements
nd = ndims(itr);                % number of tensor dimensions, i.e. order of tensor
if (ne==3), nd = 1; end         % order of tensor is 1 in case of a 3x1 or 1x3 vector

otr = itr;                      % create output tensor
otr(:) = 0;                     % fill output tensor with zeros; this way a symbolic tensor remains symbolic

iie = zeros(nd,1);              % initialise vector with indices of input tensor element
ioe = zeros(nd,1);              % initialise vector with indices of output tensor element
cne = cumprod(3*ones(nd,1))/3;  % vector with cumulative number of elements for each dimension (divided by three)

for oe = 1:ne,                                 % loop over all output elements
   ioe = mod(floor((oe-1)./cne),3)+1;          % calculate indices of current output tensor element
   for ie = 1:ne,                              % loop over all input elements
      pmx = 1;                                 % initialise product of transformation matrices
      iie = mod(floor((ie-1)./cne),3)+1;       % calculate indices of current input tensor element
      for id = 1:nd,                           % loop over all dimensions
         pmx = pmx * tmx( ioe(id), iie(id) );  % create product of transformation matrices
      end
      otr(oe) = otr(oe) + pmx * itr(ie);       % add product of transformation matrices and input tensor element to output tensor element
   end
end

% Transform matrix about Z axis
% for x = 0:pi/20:pi/2
%     trans = [cos(x)      cos(x-pi/2) 0;
%              cos(x+pi/2) cos(x)      0;
%              0           0           1];
%     N_CH = transform(C,trans);
% end