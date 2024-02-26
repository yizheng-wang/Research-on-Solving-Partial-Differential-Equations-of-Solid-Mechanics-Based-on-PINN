% input the homogenized elasticity tensor
function [s,E] = visual(CH)
% transform it to 3*3*3*3 tensor
tensor = generate(CH);
% find the E1 in 360 degree
% x = 0:pi/180:2*pi;
[a,e] = meshgrid(0:0.02*pi:2*pi, -pi/2:0.01*pi:pi/2);
E1 = zeros(size(a));
for i = 1:size(a,1)
    for j = 1:size(a,2)
        % build the transformation matrix
        trans_z = [cos(a(i,j))     -sin(a(i,j))      0;
                   sin(a(i,j))      cos(a(i,j))      0;
                   0                0                1];
        trans_y = [cos(e(i,j))      0           sin(e(i,j));
                   0                1           0;
                  -sin(e(i,j))      0           cos(e(i,j))];
        % calculate the new tensor
        N_tensor = transform(tensor,trans_y*trans_z);
        % transform the tensor to 6*6
        N_CH = ToMatrix(N_tensor);
        % calculate the E1
        E = modulus(N_CH);
        E1(i,j) = E(1);
    end
end
[x,y,z] = sph2cart(a,e,E1);
c = sqrt(x.^2+y.^2+z.^2);
s = surf(x,y,z,c);
s.EdgeColor = 'none';
s.FaceColor = 'interp';
% display(s)
end

function [E] = modulus(CH)
S = inv(CH);
E = zeros(6,1);
E(1) = 1/S(1,1);
E(2) = 1/S(2,2);
E(3) = 1/S(3,3);
E(4) = 1/S(4,4);
E(5) = 1/S(5,5);
E(6) = 1/S(6,6);
end

function C = generate(CH)
C = zeros(3,3,3,3);
for i = 1:6
    for j = 1:6
        [a,b] = change(i);
        [c,d] = change(j);
        C(a,b,c,d) = CH(i,j);
    end
end
for i = 1:3
    if (i == 3)
        j = 1;
    else
        j = i+1;
    end
    for m = 1:3
        if (m == 3)
            n = 1;
        else
            n = m+1;
        end
        C(j,i,n,m) = C(i,j,m,n);
        C(j,i,m,n) = C(i,j,m,n);
        C(i,j,n,m) = C(i,j,m,n);
        C(j,i,m,m) = C(i,j,m,m);
        C(m,m,j,i) = C(m,m,i,j);
    end
end
end

% change the index 4 5 6 to 23 31 12
function [a,b] = change(w)

if (w < 4)
    a = w;
    b = w;
else
    if (w == 4)
        a = 2;
        b = 3;
    else
        if (w == 5)
            a = 3;
            b = 1;
        else
            if (w==6)
                a = 1;
                b = 2;
            end
        end
    end
end

end

function CH = ToMatrix(C)
CH = zeros(6,6);
for i = 1:6
    for j = 1:6
        [a,b] = change(i);
        [c,d] = change(j);
        CH(i,j) = C(a,b,c,d);     
    end
end
end