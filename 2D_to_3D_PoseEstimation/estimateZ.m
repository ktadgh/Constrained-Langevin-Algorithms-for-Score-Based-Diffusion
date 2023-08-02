function Z = estimateZ(dZ, edges, Z1)

% edges = [1 2 3 4 2 6 7 2 9  1  11 12 13 1  15 16 17;...
%          2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18];


% edges = [1 2 3 4 2 6 7 2  1 10 11 12  1 14 15 16
%          2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17];

P = size(dZ,2) + 1;
F = size(dZ,1);

Z = zeros(F, P);
Z(:, 1) = Z1;

for i=1:P-1
   Z(:, edges(2,i)) = Z(:, edges(1,i)) - dZ(:, i);
end