function [X2, X3, R0] = estimatePose_v2(x, alpha)
% This code estimates 3D human pose given 2D joint annotations in an image.
% Inputs:
% x: 2D pose consistying of annotations of human joint  in an image
% alpha: the prior weight (default=2). You may want to increase/decrease if there is 
% more/less noise in the image observations
% copyright: Ijaz Akhter, MPI Tuebingen
% October 7, 2015
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

var = load('staticPose');
a = var.a;
di = var.di;
edges = var.edges;

load model8a             % loads B (dim: 3P_by_K), 
                         % vrn (dim:82_by_1), mu (dim:3_by_P)

if nargin<2
%     alpha = 3.1;
    alpha = 2;
end
trso = [1,2,3,6,10,14];
exTrso = [1 2 3 6 9 10 11 14 15];     % includes head and upper-legs
bone_Lengths = avgBone_Lengths;

L = size(edges,1);      % L = P-1
P = size(x,2);
Pt = length(exTrso);
K = size(B,2);

zInd = (3:3:3*L)';
prjInd = zeros(2*P,1);
prjInd(1:2:end) = 1:3:3*P;
prjInd(2:2:end) = 2:3:3*P;

x = x - mean(x,2)*ones(1,P);
mu = mu - mean(mu,2)*ones(1,P);
mut = mut - mean(mut,2)*ones(1,Pt);
dMu = mu(:, edges(:,1)) - mu(:, edges(:,2));

st = mean(std(x, 1, 2));
x = x/st;
dB = zeros(3*L, K);
dB(1:3:end, :) = B(3*edges(:,1)-2,:) - B(3*edges(:,2)-2,:);
dB(2:3:end, :) = B(3*edges(:,1)-1,:) - B(3*edges(:,2)-1,:);
dB(3:3:end, :) = B(3*edges(:,1),:) - B(3*edges(:,2),:);

trsoLengths = bone_Lengths(exTrso(2:end)-1);
[R0, dZt, r] = torsoBsdFixingb(x(:, exTrso), Bt, mut, Mw_t, sigw_t, trsoLengths, alpha);
 
% norm(R-R0)
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

signsAL = getSignsfromAngleLimits(dx, dZ');
indAL = signsAL~=0;
dZ(indAL) = signsAL(indAL).*abs(dZ(indAL))';

imSgnl = x2 - R(1:2,:)*mu;
imProj = imSgnl(:);

dpSgnl = dZ(indAL) - rdMu(zInd(indAL));
dpProj = dpSgnl(:);

selectIds = [];
ks = 0;
maxK = 15;

while(ks<maxK)
    
    coeffs = rB(prjInd, :)'*imProj + rdB(zInd(indAL), :)'*dpProj;
    coeffs(selectIds) = NaN;

    imax = selectBasicVector(coeffs, selectIds, edges, rB, rdB, rMu, rdMu, x2, dZ, indAL);
    if imax==0
        rBstar = rB(:, selectIds); 
        maxK = ks;
        break;
    end
    selectIds = [selectIds, imax];
    
    rBstar = rB(:, selectIds);
    rdBstar = rdB(:, selectIds);
    
    ks = ks + 1;
    omega = solveOmega(rBstar, rdBstar, rMu, rdMu, x2, dZ, indAL);
    
    sgnlRec = rBstar(prjInd, :)*omega;
    sgnlDRec = rdBstar(zInd(indAL), :)*omega;    
    imProj = imSgnl(:) - sgnlRec;
    dpProj = dpSgnl(:) - sgnlDRec;
    
    Sc = reshape(B(:, selectIds)*omega, 3,P) + mu;
    [R] = updateRotation2(x, Sc);
    
    rdMu = R*dMu;
    rdB = kron(eye(L), R)*dB;         % dim: 3L_by_N    
    rMu = R*mu;
    rB = kron(eye(P), R)*B;         % dim: 3L_by_N
end
X2b = reshape(rBstar*omega, 3, P) + rMu;

% Bstar = B(:, selectIds);
% [X2b, signs] = jointSolve4OmegaR(Bstar, mu, x, avgBone_Lengths, edges, omega, R, r);
 
w = zeros(K,1);
w(selectIds) = omega;
[X2b, signs] = jointSolve4OmegaR_v2(B, mu, x, avgBone_Lengths, edges, w, R, r);

dZ = signs'.*abs(dZ);

Z = estimateZ(dZ', edges', 0);
X2 = [x; (r*st)*Z];
X3 = (r*st)*X2b;

function imax = selectBasicVector(coeffs, selectIds, edges, rB, rdB, rMu, rdMu, x2, dZ, indAL)

P = size(x2,2);
[cf,imax] = max(abs(coeffs));
flg = true;

while(flg)
    rBstar = rB(:, [selectIds,imax]);
    rdBstar = rdB(:, [selectIds,imax]);   
    
    omega = solveOmega(rBstar, rdBstar, rMu, rdMu, x2, dZ, indAL);
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

function omega = solveOmega(rB, rdB, rMu, rdMu, x2, dZ, di)

P = size(x2,2);
L = P -1;

prjInd = zeros(2*P,1);
prjInd(1:2:end) = 1:3:3*P;
prjInd(2:2:end) = 2:3:3*P;
zInd = (3:3:3*L)';

ex = reshape((x2-rMu(1:2,:)), 2*P, 1);
bd = dZ(di) - rdMu(zInd(di));

A = [rB(prjInd, :); rdB(zInd(di),:)];
b = [ex; bd];

omega = A\b;

