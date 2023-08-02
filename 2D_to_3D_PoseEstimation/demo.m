


%% Single frame recontrsuction
load('testingSet');
S = S(:, [1:8, 10:18]);         % ground-truth 3D poses 
W = W(:, [1:8, 10:18]);         % input 2D joint annotations

P = 17;
f = 100;       % frame # you want to reconstruct, try different frames

xy = W(2*f-1:2*f, :);
X = S(3*f-2:3*f, :);
X = X - mean(X,2)*ones(1,P);

% implementation of my CVPR15 paper
[~, X3] = estimatePose(xy);     % X3 is the estimated 3D pose

% the following function gives better results than estimatePose but is slightly different than the one descirbed in the paper
[~, X4] = estimatePose_v2(xy);  


X3 = X3 - mean(X3,2)*ones(1,P);
X3 = Rs(3*f-2:3*f,:)'*X3;       % apply ground-truth alignment

load skeleton_17Pts
figure(1)
viewWnS2(xy, X, X3, edges)

%% If feet are missing
feetInd = [13,17];
xy(:, feetInd) = nan;
[~, X2] = estimatePoseMissingFeet(xy);
X2 = Rs(3*f-2:3*f,:)'*X2;

figure(2)
viewWnS2(xy, X, X2, edges)