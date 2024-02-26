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
            save('J.mat',"J");
%             det(J)
            qxyz = J\[qx;qy;qz];
            save('qxyz.mat',"qxyz");
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
            save('B.mat','B');
%             size(B)
%             B_e
            % Weight factor at this point
            weight = det(J)*ww(ii) * ww(jj) * ww(kk);
%             weight
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