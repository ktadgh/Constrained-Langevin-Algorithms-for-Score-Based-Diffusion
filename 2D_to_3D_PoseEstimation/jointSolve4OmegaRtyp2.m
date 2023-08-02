function [X2b, signs, omega, R, r] = jointSolve4OmegaRtyp2(B, dB, mu, dMu, x, lengths, edges, w0, R0, r0, indv)

P = size(x,2);
% dx = x(:, edges(:,1)) - x(:, edges(:,2));
% avgL = mean(sum(dx.^2,1))/10;
avgL = 1;

% options = optimoptions('fminunc', 'Display', 'iter', 'Algorithm', 'quasi-newton');
options = optimoptions('fminunc', 'Display', 'none', 'Algorithm', 'trust-region', 'GradObj', 'on');

y0 = [rodrigues(R0); r0; w0];
[y, fval] = fminunc(@(y)costFnc(y, x, B, mu, edges, lengths, avgL, indv), y0, options);
R = rodrigues(y(1:3));
r = y(4);
omega = y(5:end);

X2b = R*(reshape(B*omega, 3, P) + mu);
signs = sign(X2b(3,edges(:,1)) - X2b(3,edges(:,2)));

function [f,J] = costFnc(y, x, B, mu, edges, lengths, avgL, indv)

alpha = 1;
beta = 0.3;
P = size(x,2);
L = length(lengths);
k = size(B,2);

dindv = indv(2:end)-1;
r = y(1:3);
s = y(4);
w = y(5:end);
[R, dR] = rodrigues(r);

Xc = reshape(B*w, 3,P) + mu;
Xt = R*Xc;
dX = Xt(:, edges(:,1)) - Xt(:, edges(:,2));
curLengths = sum(dX.^2,1);

diff = x(:,indv) - s*Xt(1:2, indv);
reProjCost = diff(:);
lengthCost = curLengths(dindv) - lengths(dindv).^2;
% f = sum(reProjCost.^2)/avgL + beta*sum(abs(lengthCost));

% priorCost = 0;
if all(isPlausible2(dX))
    priorCost=0;
else
    priorCost = Inf;
end
% priorCost = getLogLikelihood(Xc);
f = sum(reProjCost.^2)/avgL + sum(priorCost.^2) + beta*sum(abs(lengthCost));

if nargout>1
    P = length(indv);
    x = x(:, indv);    
    indv2 = sort([3*indv-2, 3*indv-1, 3*indv]);       
    dindv2 = sort([3*dindv-2, 3*dindv-1, 3*dindv]);       
    
    dMu = mu(:, edges(:,1)) - mu(:, edges(:,2));
    dMu = dMu(:, dindv);
    mu = mu(:, indv);
    
    dMu = dMu(:);
    dB = zeros(3*L, k);
    dB(1:3:end, :) = B(3*edges(:,1)-2,:) - B(3*edges(:,2)-2,:);
    dB(2:3:end, :) = B(3*edges(:,1)-1,:) - B(3*edges(:,2)-1,:);
    dB(3:3:end, :) = B(3*edges(:,1),:) - B(3*edges(:,2),:);
    B = B(indv2,:); 
    dB = dB(dindv2,:); 
    
    sn = sign(lengthCost);    
    Sn = kron(diag(sn), eye(3));
    dfwl = 2*dB'*Sn*dMu + 2*dB'*Sn*dB*w;

    x = x(:);
    mu = mu(:);
    J = zeros(size(y));
    
    kR = kron(eye(P), R(1:2, :));
    kRRt = kron(eye(P), R(1:2, :)'*R(1:2,:));
    
    dfw1 = -2*s*B'*kR'*x + 2*s^2*B'*kRRt*(mu+B*w);    
    dfs = -2*x'*kR*(mu+B*w) + 2*s*(mu+B*w)'*kRRt*(mu+B*w);
    
    ut = reshape(mu, 3,P);
    Bt = zeros(3*k, P);
    Bt(1:3:end, :) = B(1:3:end, :)';
    Bt(2:3:end, :) = B(2:3:end, :)';
    Bt(3:3:end, :) = B(3:3:end, :)';
    wt = kron(w', eye(3));
    
    dR1 = dR(1:3:end, :);
    dR2 = dR(2:3:end, :);
    dfv_1 = 2*s^2*dR1'*(ut+wt*Bt)*(ut+wt*Bt)'*R(1,:)' - 2*s*dR1'*(ut+wt*Bt)*x(1:2:end);
    dfv_2 = 2*s^2*dR2'*(ut+wt*Bt)*(ut+wt*Bt)'*R(2,:)' - 2*s*dR2'*(ut+wt*Bt)*x(2:2:end);
    
    J(1:3) = (dfv_1 + dfv_2)/avgL;
    J(4) = dfs/avgL;
    J(5:end) = dfw1/avgL + beta*dfwl;
end
