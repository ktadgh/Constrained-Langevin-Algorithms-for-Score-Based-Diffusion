function [R, r] = estimateRotation(dMu, dx)

R1 = dx/dMu;
R1(3,:) = cross(R1(1,:), R1(2,:));
% r = (norm(R1(1,:)) + norm(R1(2,:)))/2;
if det(R1)<0
   R1(3,:) = -R1(3,:);
end
[U, D, V] = svd(R1);
R = U*V';

dx2 = R(1:2, :)*dMu;

l1 = sqrt(sum(dx.^2,1));
l2 = sqrt(sum(dx2.^2,1));
r = mean(l1./l2);