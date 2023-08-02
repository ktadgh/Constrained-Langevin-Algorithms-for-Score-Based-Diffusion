function [R, dZt, r, Xc] = torsoBsdFixingb(xt, Bt, mut, Mw, sigw, trsoLengths, alpha)

tEdges = [1 2 2 2 1 6 1 8; 2 3 4 5 6 7 8 9]';
P = size(xt,2);
L = P - 1;
zInd = (3:3:3*L)';
kt = size(Bt,2);

if nargin<7
    alpha = 2;
%     alpha = 3.1;        % it can be increased by 100-500 times
end

sigwi = inv(sigw);
D = eye(kt) - Mw;
ApA = alpha*D'*sigwi*D;
[U,D0,V] = svd(ApA);
Ap = U*sqrt(D0);

% mut = mut - mean(mut,2)*ones(1,P);      % not necessary, mean is already zero
xt = xt - mean(xt,2)*ones(1,P);

[R, r] = estimateRotation(mut, xt);
rMu = R*mut;
rB = kron(eye(P), R)*Bt;         % dim: 3L_by_N 

omega = solveOmega(rB, rMu, xt/r, Ap);


%%

dx = xt(:, tEdges(:,1)) - xt(:, tEdges(:,2));
% maxr = max(sqrt(sum(dx.^2,1))./trsoLengths);
% avgL = mean([r,maxr])^2/10;
% avgL = mean(sum(dx.^2,1))/10;
avgL = 1;

% options = optimoptions('fminunc', 'Display', 'iter');
% options = optimoptions('fminunc', 'Display', 'none', 'Algorithm', 'quasi-newton');
options = optimoptions('fminunc', 'Display', 'none', 'Algorithm', 'trust-region', 'GradObj', 'on');

% y0 = rodrigues(R); 
% [y, fval] = fminunc(@(y)costFnc(y, xt, Bt, mut, Ap), y0);
% R = rodrigues(y);

y0 = [rodrigues(R); r; omega];
[y, fval] = fminunc(@(y)costFnc2(y, xt, Bt, mut, Ap, trsoLengths, tEdges, avgL, alpha), y0, options);
R = rodrigues(y(1:3));
r = y(4);

rMu = R*mut;
rB = kron(eye(P), R)*Bt;         % dim: 3L_by_N 
omega = solveOmega(rB, rMu, xt/r, Ap);
Xc = reshape(rB*omega, 3,P) + rMu;

%%
dXt = Xc(:, tEdges(:,1)) - Xc(:, tEdges(:,2));
dZt = r*dXt(3,:);


function [f, J] = costFnc2(y, x, B, mu, Ap, trsoLengths, tEdges, avgL, alpha)

% beta = 0.3; 
beta = 3;
P = size(x,2);
L = length(trsoLengths);
k = size(B,2);

r = y(1:3);
s = y(4);
w = y(5:end);
[R, dR] = rodrigues(r);

Xc = reshape(B*w, 3,P) + mu;
Xt = R*Xc;
dX = Xt(:, tEdges(:,1)) - Xt(:, tEdges(:,2));
curLengths = sum(dX.^2,1);

diff = x - s*Xt(1:2, :);
reProjCost = diff(:);
priorCost = Ap*w;
lengthCost = curLengths - trsoLengths.^2;

% f = sum(reProjCost.^2)/avgL + alpha*sum(priorCost.^2) + beta*sum(lengthCost.^2);
f = sum(reProjCost.^2)/avgL + alpha*sum(priorCost.^2) + beta*sum(abs(lengthCost));

if nargout>1
    
    dMu = mu(:, tEdges(:,1)) - mu(:, tEdges(:,2));
    dMu = dMu(:);
    dB = zeros(3*L, k);
    dB(1:3:end, :) = B(3*tEdges(:,1)-2,:) - B(3*tEdges(:,2)-2,:);
    dB(2:3:end, :) = B(3*tEdges(:,1)-1,:) - B(3*tEdges(:,2)-1,:);
    dB(3:3:end, :) = B(3*tEdges(:,1),:) - B(3*tEdges(:,2),:);
    
    sn = sign(lengthCost);    
    Sn = kron(diag(sn), eye(3));
    dfwl = 2*dB'*Sn*dMu + 2*dB'*Sn*dB*w;

    x = x(:);
    mu = mu(:);
    J = zeros(size(y));
    
    kR = kron(eye(P), R(1:2, :));
    kRRt = kron(eye(P), R(1:2, :)'*R(1:2,:));
    
    dfw1 = -2*s*B'*kR'*x + 2*s^2*B'*kRRt*(mu+B*w);
    dfw2 = 2*(Ap'*Ap)*w;
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
    J(5:end) = dfw1/avgL + alpha*dfw2 + beta*dfwl;
end

function f = costFnc(r, x, B, mu, Ap)

alpha = 0.02;
P = size(x,2);

R = rodrigues(r);
rB = kron(eye(P), R(1:2,:))*B;
rMu = R(1:2,:)*mu;

A = rB;
ex = reshape(x-rMu, 2*P, 1);
w = A\ex;

reProjCost = x(:)-A*w-rMu(:);
priorCost = alpha*Ap*w;
f = sum(reProjCost.^2) + sum(priorCost.^2);

function omega = solveOmega(rB, rMu, x, Ap)

P = size(x,2);
K = size(rB,2);

prjInd = zeros(2*P,1);
prjInd(1:2:end) = 1:3:3*P;
prjInd(2:2:end) = 2:3:3*P;

ex = reshape((x-rMu(1:2,:)), 2*P, 1);
A = [rB(prjInd, :); Ap];
b = [ex; zeros(K,1)];
omega = A\b;
