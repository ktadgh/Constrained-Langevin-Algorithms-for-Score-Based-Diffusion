function [R, r] = updateRotation2(W, Sc)

R = W/Sc;
R(3,:) = cross(R(1,:), R(2,:));
r = (norm(R(1,:)) + norm(R(2,:)))/2;

if det(R)<0
    R(3,:) = -R(3,:);
end
[U, ~, V] = svd(R);
R = U*V';