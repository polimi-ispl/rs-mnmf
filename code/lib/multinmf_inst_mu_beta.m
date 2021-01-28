function [Q, W, H, cost] = multinmf_inst_mu_beta(V, beta, n_iter, Q, W, H, part, switch_Q, switch_W, switch_H)

% Multichannel NMF minimizing Itakura-Saito divergence through multiplicative updates
% with linear instantaneous mixing
%
% [Q, W, H, cost] = multinmf_inst_mu(V, n_iter, Q, W, H, part, switch_Q, switch_W, switch_H)
%
% Input:
%   - V: positive matrix data       (F x N x n_c)
%   - beta: the beta divergence value \in [0,2]
%   - n_iter: number of iterations
%   - init_Q: mixing matrix         (n_c x n_s)
%   - init_W: basis                 (F x K)
%   - init_H: activation coef.      (K x N)
%   - part : component indices
%   - switch_W, switch_H, switch_Q: (opt) switches (0 or 1) (def = 1)
%
% Output:
%   - Estimated Q, W and H
%   - Cost through iterations betw. data power and fitted variance.
%
%
% This software is distributed under the terms of the GNU Public License
% version 3 (http://www.gnu.org/licenses/gpl.txt)
%
% This code is based on
%
% A. Ozerov and C. Fevotte,
% "Multichannel nonnegative matrix factorization in convolutive mixtures for audio source separation,"
% IEEE Trans. on Audio, Speech and Lang. Proc. special issue on Signal Models and Representations
% of Musical and Environmental Sounds, vol. 18, no. 3, pp. 550-563, March 2010.
% Available: http://www.irisa.fr/metiss/ozerov/Publications/OzerovFevotte_IEEE_TASLP10.pdf
%
% The code has been adapted to the beta-divergence cost function by
% J. J. Carabias-Orti 
%
% If you use this code please cite this paper
% M. Pezzoli, J. J. Carabias-Orti, M. Cobos, F. Antonacci,and A. Sarti. 
% "Ray-space-based multichannel nonnegative matrix factorization for audio 
% source separation". IEEE Signal Processing Letters (2021).
% doi: 10.1109/LSP.2021.3055463


if nargin < 8 || isempty(switch_Q)
    switch_Q = 1;
end

if nargin < 9 || isempty(switch_W)
    switch_W = 1;
end

if nargin < 10 || isempty(switch_H)
    switch_H = 1;
end

[F,N,n_c] = size(V);
n_s = size(Q,2);

%% Definitions %%
V_ap    = zeros(F,N,n_c);
cost    = zeros(1,n_iter);

%% Compute app. variance structure V_ap %%
for j=1:n_s
    P_j = W(:,part{j}) * H(part{j},:);    
    for i=1:n_c
        V_ap(:,:,i) = V_ap(:,:,i) + Q(i,j) .* P_j;
    end
end
V(V == 0) = eps;
if beta == 1
    cost(1) = sum(V(:).*log((V(:)./(V_ap(:)+eps))+eps)-V(:)+V_ap(:));
elseif beta==0
    cost(1) = sum((V(:)./(V_ap(:)+eps)) - log((V(:)./(V_ap(:)+eps))+eps)-1);
else
    cost(1) = 1/(beta*(beta-1)) * sum( V(:).^beta + ((beta-1)*V_ap(:).^beta) - (beta*V(:).*(V_ap(:).^(beta-1))) );
end

for iter = 2:n_iter
    %%% Update Q %%%
    if switch_Q
        for j=1:n_s  
            P_j = W(:,part{j}) * H(part{j},:);            
            for i=1:n_c
                Q_old  = Q(i,j);
                Q(i,j) = Q(i,j) * sum(sum(V_ap(:,:,i).^(beta-2).*P_j.*V(:,:,i))) / sum(sum(V_ap(:,:,i).^(beta-1) .* P_j));
                V_ap(:,:,i) = V_ap(:,:,i) + (Q(i,j)-Q_old) .* P_j;                
            end
        end
    end

    %%% Update W %%%
    if switch_W
        for j=1:n_s
            Kj   = length(part{j});
            Wnum = zeros(F,Kj);
            Wden = zeros(F,Kj);
            for i=1:n_c
                Wnum = Wnum + Q(i,j) * ((V_ap(:,:,i).^(beta-2) .* V(:,:,i)) * H(part{j},:)');
                Wden = Wden + Q(i,j) * (V_ap(:,:,i).^(beta-1) * H(part{j},:)');
            end
            
            Wj_old = W(:,part{j});
            W(:,part{j}) = W(:,part{j}) .* (Wnum./Wden);

            for i=1:n_c
                V_ap(:,:,i) = V_ap(:,:,i) + Q(i,j) * ((W(:,part{j})-Wj_old)*H(part{j},:));
            end
        end
    end

    %%% Update H %%%
    if switch_H
        for j=1:n_s
            Kj   = length(part{j});
            Hnum = zeros(Kj,N);
            Hden = zeros(Kj,N);
            for i=1:n_c
                Hnum = Hnum + (Q(i,j) * W(:,part{j})') * (V_ap(:,:,i).^(beta-2) .* V(:,:,i));
                Hden = Hden + (Q(i,j) * W(:,part{j})') * V_ap(:,:,i).^(beta-1);
            end
            
            Hj_old = H(part{j},:);
            H(part{j},:) = H(part{j},:) .* (Hnum./Hden);

            for i=1:n_c
                V_ap(:,:,i) = V_ap(:,:,i) + Q(i,j) * (W(:,part{j})*(H(part{j},:)-Hj_old));
            end
            
        end

    end

    if beta == 1
        cost(iter) = sum(V(:).*log((V(:)./(V_ap(:)+eps))+eps)-V(:)+V_ap(:));
    elseif beta==0
        cost(iter) = sum((V(:)./(V_ap(:)+eps)) - log((V(:)./(V_ap(:)+eps))+eps)-1);
    else
        cost(iter) = 1/(beta*(beta-1)) * sum( V(:).^beta + ((beta-1)*V_ap(:).^beta) - (beta*V(:).*(V_ap(:).^(beta-1))) );
    end

    %%% Normalize %%%    
    
    %% Scale Q / W %%
    if switch_Q && switch_W               
        scale = sum(Q,1);
        Q = Q ./ repmat(scale,n_c,1);
        for j=1:n_s
            W(:,part{j}) = W(:,part{j}) * scale(j);
        end
    end
    
    %% Scale W / H %%
    if switch_W && switch_H
        scale = sum(W,1);
        W = W ./ repmat(scale ,F,1);
        H = H .* repmat(scale',1,N);
    end

end
fprintf('MU %d cost: %.4f -> %.4f\n', n_iter, cost(2), cost(iter));


