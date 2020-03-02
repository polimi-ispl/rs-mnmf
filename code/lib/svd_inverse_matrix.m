function [G_INV, outParam] = svd_inverse_matrix(G,regParams)
% Copyright 2020 Mirco Pezzoli
% (mirco.pezzoli -at- polimi.it)
%
% This software is distributed under the terms of the GNU Public License
% version 3 (http://www.gnu.org/licenses/gpl.txt)
%
% If you use this code please cite this paper
%

method = regParams.method;
q      = regParams.nCond;

if strcmp(method,'truncated_svd')
    
    [U,E,V] = svd(G);
    N = size(G,1);
    M = size(G,2);
    
    m = min(M,N);

    sing_values = diag(E(1:m,1:m));
    max_sv = sing_values(1);
    k = length(find(sing_values >= max_sv/q));
    E2 = diag(1./sing_values(1:k));
    E2_inv = zeros(M,N);
    E2_inv(1:k,1:k) = E2;
    G_INV  = (V * E2_inv * U');
    numsv = k;
    outParam = numsv; % number of singular values
    
elseif strcmp(method,'tikhonov')
    
    [~,E,~] = svd(G);
    N = size(G,1);
    M = size(G,2);
    m = min(size(E));
    
    sing_values = diag(E(1:m,1:m));
    max_sv = sing_values(1);
    min_sv=sing_values(end);
    tik=0.5;%(max_sv^2-min_sv^2*q^2)/(q^2-1);
    
    if N < M
        G_INV = G'*inv(G*G' + tik*eye(N));
                outParam = tik; % regularization parameter

    elseif max_sv/min_sv<=q
        G_INV  = (G'*G)\G';  
        outParam = 0;
    else    
        G_INV  = (G'*G+tik*eye(M))\G';
        outParam = tik; % regularization parameter
    end
    
end
