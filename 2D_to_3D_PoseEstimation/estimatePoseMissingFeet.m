function [X2, X3, R0] = estimatePoseMissingFeet(x, alpha)
% This code estimates 3D human pose given 2D joint annotations in an image, if feet are missing.
% For details on how does it work, please see our paper in CVPR2015, "Pose-Conditioned 
% Joint Angle Limits for 3D Human Pose Reconstruction".
% Inputs:
% x: 2D pose consistying of annotations of human joint  in an image
% alpha: the prior weight (default=2). You may want to increase/decrease if there is 
% more/less noise in the image observations
% copyright: Ijaz Akhter, MPI Tuebingen
% May 22, 2015
%

global jmp chlds prnts angleSprd sepPlane E2 bounds a di % used in isplausible2

% var = load('jointAngleModel_v2');
var = load('jointAngleModel');
jmp = var.jmp;
chlds = var.chlds;
prnts = var.prnts;
angleSprd = var.angleSprd;
sepPlane = var.sepPlane;
E2 = var.E2;
bounds = var.bounds;

indv = [1:12,14:16];
var = load('staticPose');
a = var.a;
di = var.di;
edges = var.edges;

load model8a             % loads B (dim: 3P_by_K), 
                         % vrn (dim:82_by_1), mu (dim:3_by_P)

if nargin<2
    alpha = 2;
end
trso = [1,2,3,6,10,14];
exTrso = [1 2 3 6 9 10 11 14 15];     % includes head and upper-legs
bone_Lengths = avgBone_Lengths;

L = size(edges,1);      % L = P-1
P = size(x,2);
Pt = length(exTrso);
K = size(B,2);

x(:,indv) = x(:,indv) - mean(x(:,indv),2)*ones(1,length(indv));

mu = mu - mean(mu,2)*ones(1,P);
mut = mut - mean(mut,2)*ones(1,Pt);
dMu = mu(:, edges(:,1)) - mu(:, edges(:,2));

st = mean(std(x(:,indv), 1, 2));
x = x/st;
dB = zeros(3*L, K);
dB(1:3:end, :) = B(3*edges(:,1)-2,:) - B(3*edges(:,2)-2,:);
dB(2:3:end, :) = B(3*edges(:,1)-1,:) - B(3*edges(:,2)-1,:);
dB(3:3:end, :) = B(3*edges(:,1),:) - B(3*edges(:,2),:);

trsoLengths = bone_Lengths(exTrso(2:end)-1);
[R0, dZt, r, Xt] = torsoBsdFixingb(x(:, exTrso), Bt, mut, Mw_t, sigw_t, trsoLengths, alpha);
R = R0;
% pause;

x2 = x/r;
dx = x2(:, edges(:,1)) - x2(:, edges(:,2));

dZSqr = bone_Lengths.^2 - sum(dx.^2, 1);
dZSqr(dZSqr<0) = 0;
dZ = sqrt(dZSqr)';

rdMu = R*dMu;
rdB = kron(eye(L), R)*dB;         % dim: 3L_by_N 

rMu = R*mu;
rB = kron(eye(P), R)*B;         % dim: 3L_by_N 

%%
ind = [1:3, 5,7];       % torso points
dZ(trso(2:end)-1) = sign(dZt(ind)').*dZ(trso(2:end)-1);

signsAL = getSignsfromAngleLimits(dx, dZ',2);
% signsALold = getSignsfromAngleLimitsOld(dx, dZ');
% dX = X(:, edges(:,1)) - X(:, edges(:,2));
% signsg = sign(dX(3,:));

indAL = signsAL~=0;
dZ(indAL) = signsAL(indAL).*abs(dZ(indAL))';

I3 = zeros(3*length(indv),1);
I3(1:3:end) = 3*indv-2;
I3(2:3:end) = 3*indv-1;
I3(3:3:end) = 3*indv;

zInd = (3:3:3*L)';
prjInd = zeros(2*length(indv),1);
prjInd(1:2:end) = 3*indv-2;
prjInd(2:2:end) = 3*indv-1;


imSgnl = x2(:,indv) - R(1:2,:)*mu(:,indv);
imProj = imSgnl(:);

dpSgnl = dZ(indAL) - rdMu(zInd(indAL));
dpProj = dpSgnl(:);

% Ap = [];
% rBstar = [];             % Selected basis
% rdBstar = [];
selectIds = [];
ks = 0;
maxK = 15;
% alpha = 0.0001;
while(ks<maxK)
    
    coeffs = rB(prjInd, :)'*imProj + rdB(zInd(indAL), :)'*dpProj;
    coeffs(selectIds) = NaN;
%     [cf,imax] = max(abs(coeffs));
    imax = selesBasicVector(coeffs, selectIds, edges, rB, rdB, rMu, rdMu, x2, dZ, indAL, indv, prjInd);
    if imax==0
        rBstar = rB(:, selectIds);       
        maxK = ks;
        break;
    end
    selectIds = [selectIds, imax];
    % check if this is valid!!!!!
    
    rBstar = rB(:, selectIds);
    rdBstar = rdB(:, selectIds);
    
    ks = ks + 1;
    omega = solveOmega(rBstar, rdBstar, rMu(:,indv), rdMu, x2(:,indv), dZ, indAL, prjInd);
    
    sgnlRec = rBstar(prjInd, :)*omega;
    sgnlDRec = rdBstar(zInd(indAL), :)*omega;    
    imProj = imSgnl(:) - sgnlRec;
    dpProj = dpSgnl(:) - sgnlDRec;
    
%     X2b = reshape(rBstar*omega, 3, P) + rMu;    
%     viewWnS2(x2, X2b(1:2,:), X/r, X2b);

    Sc = reshape(B(:, selectIds)*omega, 3,P) + mu;
%     Sc = imposeLengths(Sc, edges, avgBone_Lengths);
    [R] = updateRotation2(x(:,indv), Sc(:,indv));
    
    rdMu = R*dMu;
    rdB = kron(eye(L), R)*dB;         % dim: 3L_by_N    
    rMu = R*mu;
    rB = kron(eye(P), R)*B;         % dim: 3L_by_N

%     err(ks) = norm(reProj);
%     is(ks) = imax;
end
X2b = reshape(rBstar*omega, 3, P) + rMu;
dX = X2b(:, edges(:,1)) - X2b(:, edges(:,2));

Bstar = B(:, selectIds);
dBstar = dB(:, selectIds);

% if all(isPlausible2(dX))        % not necessary!!! 
%     [X2b, signs] = jointSolve4OmegaRtyp2(Bstar, dBstar, mu, dMu, x, avgBone_Lengths, edges, omega, R, r, indv);
% else
%     R1 = estimateRotation(mut, x(:, exTrso));
%     [X2b, signs] = jointSolve4OmegaRtyp2(Bstar, dBstar, mu, dMu, x, avgBone_Lengths, edges, zeros(maxK, 1), R1, r, indv);
% end
[X2b, signs] = jointSolve4OmegaRtyp2(Bstar, dBstar, mu, dMu, x, avgBone_Lengths, edges, omega, R, r, indv);

dZ = signs'.*abs(dZ);
Z = estimateZ(dZ', edges', 0);
% x = x - x(:,1)*ones(1,P);
X2 = [x; (st*r)*Z];
X3 = (st*r)*X2b;

% X3 = r*R*Sc;
% % X3 = imposeLengths(X3, edges, r*bone_Lengths);

function imax = selesBasicVector(coeffs, selectIds, edges, rB, rdB, rMu, rdMu, x2, dZ, indAL, indv, prjInd)

P = size(x2,2);
[cf,imax] = max(abs(coeffs));
flg = true;

while(flg)
    rBstar = rB(:, [selectIds,imax]);
    rdBstar = rdB(:, [selectIds,imax]);   
    
    omega = solveOmega(rBstar, rdBstar, rMu(:,indv), rdMu, x2(:,indv), dZ, indAL, prjInd);
    Xh = reshape(rBstar*omega, 3,P) + rMu;
    dX = Xh(:, edges(:,1)) - Xh(:, edges(:,2));
    if all(isPlausible2(dX))
        flg = false;
    else
        coeffs(imax) = NaN;
        [cf,imax] = max(abs(coeffs));
        if isnan(cf)
            imax=0;
            flg = false;
        end
    end
end

function omega = solveOmega(rB, rdB, rMu, rdMu, x2, dZ, di, prjInd)

P = size(x2,2);
L = length(dZ);
K = size(rB,2);

% prjInd = zeros(2*P,1);
% prjInd(1:2:end) = 1:3:3*P;
% prjInd(2:2:end) = 2:3:3*P;
zInd = (3:3:3*L)';

ex = reshape((x2-rMu(1:2,:)), 2*P, 1);
bd = dZ(di) - rdMu(zInd(di));

% A = [rB(prjInd, :); Ap; rdB(zInd(di),:)];
% b = [ex; zeros(K,1); bd];

A = [rB(prjInd, :); rdB(zInd(di),:)];
b = [ex; bd];

omega = A\b;


function [omega, signs] = solveOmega2(rB, rdB, rMu, rdMu, x2, dZ, Ap, di, H)

P = size(x2,2);
L = P -1;
K = size(rB,2);

prjInd = zeros(2*P,1);
prjInd(1:2:end) = 1:3:3*P;
prjInd(2:2:end) = 2:3:3*P;
zInd = (3:3:3*L)';

li = ~di;
l = sum(di);            % # of bones with known signs
n = sum(li);            % # of the rest of the bones

options = optimoptions(@lsqlin,'Display','off');
lb = [-Inf(K,1); -ones(n,1)];
ub = [Inf(K,1); ones(n,1)];

ex = reshape((x2-rMu(1:2,:)), 2*P, 1);
bp = zeros(K,1);

bd = dZ(di) - rdMu(zInd(di));
A = [[rB(prjInd, :), zeros(2*P,n) ]; [Ap, zeros(K,n)]; ...
    [rdB(zInd(li),:), -diag(abs(dZ(li)))]; ...
    [rdB(zInd(di),:), zeros(l,n)] ];
b = [ex; bp; -rdMu(zInd(li)); bd];

% bd = H(:,di)*(dZ(di) - rdMu(zInd(di)));
% A = [[rB(prjInd, :), zeros(2*P,n) ]; [Ap, zeros(K,n)]; ...
%     [H(:,li)*rdB(zInd(li),:), -H(:,li)*diag(abs(dZ(li)))]; ...
%     [H(:,di)*rdB(zInd(di),:), zeros(P,n)] ];
% b = [ex; bp; -H(:,li)*rdMu(zInd(li)); bd];

% w0 = A\b;
w0 = zeros(K+n,1);
A0 = [rB(prjInd, :); Ap];
b0 = [ex; bp];
w0(1:K) = A0\b0;
w = lsqlin(A,b,[],[],[],[], lb, ub, w0, options);

omega = w(1:K);
signs = sign(w(K+1:K+n));


function X = imposeLengths(X, edges, bone_Lengths)

P = size(X,2);
Xm = mean(X, 2);

dX = X(:, edges(:,1)) - X(:, edges(:,2));
curL = sqrt(sum(dX.^2, 1));

scale = ones(3,1)*(bone_Lengths./curL);
dX = scale.*dX;

X = estimateZ(dX, edges', [0;0;0]);
X = X + (Xm - mean(X,2))*ones(1,P);
