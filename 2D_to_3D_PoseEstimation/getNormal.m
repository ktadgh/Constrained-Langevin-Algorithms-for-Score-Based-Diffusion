function [n, flg] = getNormal(x1, a, x)

nth = 1e-4;
if norm(x-a)<nth || norm(x+a)<nth   % x and a are parallel
    n = cross(x, x1);
    flg = true;
else
    n = cross(a, x);
    flg = false;
end
n = n/norm(n);
end