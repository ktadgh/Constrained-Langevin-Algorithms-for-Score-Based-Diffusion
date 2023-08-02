function dSl = global2local(dS)
% convert relative coordinates (with origin at the parent) to local coordinates
% dSl is of the same size as dS and will contain have the same torso
% coordinates (left/right shoulder and hip and belly)

global prnts chlds chldsT di a
% load staticPose         % load di, a, Si, edges


nprts = length(chlds);      % excluding torso
shldr = dS(:, 5) - dS(:, 2);
hip = dS(:, 13) - dS(:, 9);

dSl = dS;
for i=1:nprts
    
    if ismember(chlds(i), chldsT)
        if i==1 || i==3 || i==5
            u = shldr;
        else
            u = hip;
        end
        u = u/norm(u);
        v = dS(:,1);
        v = v/norm(v);
    else
        u = dS(:, prnts(i));
        u = u/norm(u);
        [v] = getNormal(R*di(:, prnts(i)), R*a, u);       % v is perpendicular to u
    end
    w = cross(u,v);
    w = w/norm(w);
    R = gramschmidt([u, v, w]);
    
%     if typ==1
%         R = gramschmidt([u, v, w]);
%     elseif typ==2
%         R = gramschmidt([w, u, v]);
%     elseif typ==3
%         R = gramschmidt([v, w, u]);
%     elseif typ==4
%         R = gramschmidt([u, -w, v]);
%     elseif typ==5
%         R = gramschmidt([-w, v, u]);
%     else
%         R = gramschmidt([v, u, -w]);
%     end
    
    dSl(:, chlds(i)) = R'*dS(:, chlds(i));
end

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