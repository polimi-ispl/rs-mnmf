function [G_INV, outParam] = svdinversematrix(G,tik)
%% svdinversematrix
% This function compute the regularized inverse of the matrix G using
% tikhonov.
%
% Params:
%   - G: the matrix to be inverted
%   - tik: tikhonov parameter
% 
% Returns: 
%   - G_INV: the regularized inverse matrix
%   - outParam: the tikhonov parameter
%
% Copyright 2020 Mirco Pezzoli
% (mirco.pezzoli -at- polimi.it)
%
% This software is distributed under the terms of the GNU Public License
% version 3 (http://www.gnu.org/licenses/gpl.txt)
%
% Based on:
% Yagle, Andrew E. "Regularized matrix computations."
% matrix 500 (2005): 10.

[~,E,~] = svd(G);
N = size(G,1);
M = size(G,2);
m = min(size(E));

sing_values = diag(E(1:m,1:m));
max_sv = sing_values(1);
min_sv=sing_values(end);

if N < M
    G_INV = G'*inv(G*G' + tik*eye(N));
    outParam = tik; % regularization parameter
elseif max_sv/min_sv<=tik
    G_INV  = (G'*G)\G';
    outParam = 0;
else
    G_INV  = (G'*G+tik*eye(M))\G';
    outParam = tik; % regularization parameter
end

end
