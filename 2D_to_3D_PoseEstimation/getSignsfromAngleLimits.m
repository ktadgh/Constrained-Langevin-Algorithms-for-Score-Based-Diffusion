function signs = getSignsfromAngleLimits(dx, dZ, typ)
% Assumption: torso signs has been determined already

if nargin<3
    typ=1;
    prtsInd = {[3,4], [6,7], 8, 10:12, 14:16};
else
    prtsInd = {[3,4], [6,7], 8, 10:11, 14:15};
end
signs = zeros(1,16);
    
for i=1:5
    N = length(prtsInd{i});    
    signPerm = npermutek([1 -1],N);
    flags = false(size(signPerm));
    
    dZi = dZ;
    for j=1:size(signPerm,1)
        dZi(prtsInd{i}) = signPerm(j,:).*abs(dZi(prtsInd{i}));
        dS = [dx; dZi];        

        if typ==1       % if  there are feet in input data
            flags(j,:) = isPlausible2(dS, i);
        else
            flags(j,:) = isPlausible3(dS, i);
        end
    end
    ind = all(flags, 2);
    lftOvrPerm = signPerm(ind, :);

    if isempty(lftOvrPerm)
        continue;
    end
    for j=1:N
        if all(lftOvrPerm(:,j)==1)
            signs(prtsInd{i}(j)) = 1;
        elseif all(lftOvrPerm(:,j)==-1)
            signs(prtsInd{i}(j)) = -1;
        end
    end
end