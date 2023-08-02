function [flags] = isPlausible3(dS, prtN)
% dS consists of the relative coordinates, origin for each joint is its
% parent.
% PrtN: part #, 1:5 for left-arm, right-arm, head, left-lef, and right-head

global jmp chlds prnts angleSprd sepPlane E2 bounds chldsT  di a

% var = load('jointAngleLimits7_1');
% jmp = var.jmp;
% 
% chlds = var.chlds;
% prnts = var.prnts;
% 
% angleSprd = var.angleSprd;
% sepPlane = var.sepPlane;
% E2 = var.E2;
% bounds = var.bounds;
% 
% var = load('staticPose');
% a = var.a;
% di = var.di;

chldsT = [3 6 8 10 14];     % torso's child
nprts = length(chlds);      % excluding torso

if nargin<2
    flags = true(1,size(dS,2));
    angles = zeros(2,size(dS,2));
    dSl = global2local(dS);     % convert relative to local coordinates 
    
    for i=1:nprts
        chldB = dSl(:, chlds(i));           % the bone to validate
        [th, phi, r] = cart2sph(chldB(1), chldB(2), chldB(3));
        chldB = chldB/r;
        th = radtodeg(th);
        phi = radtodeg(phi);
        t_j = floor((th+180)/jmp + 1);
        p_j = floor((phi+90)/jmp + 1);
        angles(:, chlds(i)) = [t_j; p_j];
        
        if ismember(chlds(i), chldsT)
            if ~angleSprd{i}(t_j, p_j)
                flags(chlds(i)) = false;
            end            
        else
            t_p = angles(1,prnts(i));
            p_p = angles(2,prnts(i));
            
            v = squeeze(sepPlane{i}(t_p,p_p,:));
            v = v/norm(v(1:3));
           
            if any(isnan(v)) || v'*[chldB;1]>0
                flags(chlds(i)) = false;
            else
                e1 = v(1:3);
                e2 = squeeze(E2{i}(t_p,p_p,:));
                T = gramschmidt([e1,e2,cross(e1, e2)]);
                bnd = squeeze(bounds{i}(t_p,p_p,:));
                
                u = T(:,2:3)'*chldB;
                if u(1)<bnd(1) || u(1)>bnd(2) || u(2)<bnd(3) || u(2)>bnd(4)
                    flags(chlds(i)) = false;
                end
            end
        end
    end
else
    prtsInd = {[1,2], [3,4], 5,  6:7, 9:10};
    prtsI = prtsInd{prtN};
    N = length(prtsI);
    flags = true(1,N);    
    
    v = dS(:,1);            % torso
    if prtN>=1 && prtN<=3
        u = dS(:, 5) - dS(:, 2);    % shoulder
    else
        u = dS(:, 13) - dS(:, 9);   % hip
    end
    u = u/norm(u);
    v = v/norm(v);

    for j=1:N
        i = prtsI(j);
        if j>1
            u = dS(:, prnts(i));
            u = u/norm(u);            
            v = getNormal(R*di(:, prnts(i)), R*a, u);       % v is perpendicular to u
        end
        w = cross(u,v);
        w = w/norm(w);
        R = gramschmidt([u, v, w]);
        
        chldB = R'*dS(:, chlds(i));
        [th, phi, r] = cart2sph(chldB(1), chldB(2), chldB(3));
        chldB = chldB/r;
        
        th = radtodeg(th);
        phi = radtodeg(phi);
        if j==1
            t_j = floor((th+180)/jmp + 1);
            p_j = floor((phi+90)/jmp + 1);
            if ~angleSprd{i}(t_j, p_j)
                flags(j) = false;
            end
        else
            %%            
            v = squeeze(sepPlane{i}(t_j,p_j,:));
            v = v/norm(v(1:3));
            
            if any(isnan(v)) || v'*[chldB;1]>0
                flags(j) = false;
            else
                e1 = v(1:3);
                e2 = squeeze(E2{i}(t_j,p_j,:));
                T = gramschmidt([e1,e2,cross(e1, e2)]);
                bnd = squeeze(bounds{i}(t_j,p_j,:));
                                
                u = T(:,2:3)'*chldB;
                if u(1)<bnd(1) || u(1)>bnd(2) || u(2)<bnd(3) || u(2)>bnd(4)
                    flags(j) = false;
                end
            end
            %%
            t_j = floor((th+180)/jmp + 1);
            p_j = floor((phi+90)/jmp + 1);
        end
    end
end