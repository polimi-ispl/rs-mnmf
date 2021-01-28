function [estimateImage, Q, basisF, activationF, xRaySpace, psi, invPsi, initQ, cost] = rayspacenmf(micSTFT, muBar, D, nuBar, sigma, fAx, d, nMic, c, sourceN, nBasisSource,...
    nIter, tik, init, beta)
% rayspacenmf
% This function performs the source separation using the Ray-Space-Based
% Multichannel Nonnegative  Matrix factorization.
% The Ray Space representation of the array signal is computed and the
% source images in the Ray Space are computed by the MNMF.
%
% Params:
%   - micSTFT: The multichannel STFTs of size freqN x timeN x micN.
%   - muBar: The sampling of the mu-axis (plane wave directions).
%   - D: Length of mu axis.
%   - nuBar: The sampling of the nu-axis (subarray position).
%   - sigma: Gaussian window standard deviation.
%   - fAx: Frequency axis.
%   - d: Distance between two consecutive microphones.
%   - nMic: Number of microphones.                   
%   - c: Sound speed.
%   - sourceN: Number of sources.
%   - nBasisSource: Number of basis function used  in the NMF for each
%   source.
%   - nIter: Number of MNMF iterations.
%   - tik: Tikhonov parameter for the inverse computation.
%   - init: Struct that contains the initialization of the algorithm if
%   empty the initialization is computed.
%   - beta: value for the beta divergence if empy IS divergence is used
%
% Returns:
%   - estimateImage: The estimated source images in the Ray Space.
%   - Q: The Ray Space mixing coefficients.
%   - basisF: The basis functions of the sources.
%   - activationF: The activation functions of the sources.
%   - xRaySpace: The Ray-Space-transformed data.
%   - psi: The Ray Space tranform matrix
%   - invPsi: The inverse Ray Space transform matrix.
%   - initQ: The initialization of the Ray Space mixing coeffients.
%
% Copyright 2020 Mirco Pezzoli
% (mirco.pezzoli -at- polimi.it)
%
% This software is distributed under the terms of the GNU Public License
% version 3 (http://www.gnu.org/licenses/gpl.txt)
%
% If you use this code please cite this paper
% M. Pezzoli, J. J. Carabias-Orti, M. Cobos, F. Antonacci,and A. Sarti. 
% "Ray-space-based multichannel nonnegative matrix factorization for audio 
% source separation". IEEE Signal Processing Letters (2021).
% doi: 10.1109/LSP.2021.3055463

fLen = size(micSTFT, 1);
tLen = size(micSTFT, 2);
nBasis = nBasisSource * sourceN;
sourceNMFidx = cell(1,sourceN);
for iSrc = 1:sourceN
    sourceNMFidx{iSrc} = (1:nBasisSource) + (iSrc-1)*nBasisSource;
end

%% Ray space projection
fprintf('RaySpace transform...\n');
% Ray space transformation matrix
psi = rayspacetransformmatrix(fAx,c,d,nMic,muBar,D,nuBar,sigma);
I = size(psi, 1);        % Number of Ray space data points

invPsi = zeros(size(psi,2),size(psi,1),fLen);
for ff = 2:fLen
    invPsi(:,:,ff) = regularizedinversematrix(psi(:,:,ff), tik);
end

% Ray-Space-transformed data
xRaySpace = spatialfilter(micSTFT, psi,false);

%% MNMF of rayspace data
fprintf('RS-MCNMF...\n');
if isempty(init)
    % Parameters initialization
    psdMix = 0.5 * (mean(abs(xRaySpace(:,:,1)).^2 + abs(xRaySpace(:,:,2)).^2, 2));
    initA = 0.5 * (1.9 * abs(randn(I, sourceN)) + ...
        0.1 * ones(I, sourceN));
    % W is intialized so that its enegy follows mixture PSD
    initW = 0.5 * (abs(randn(fLen,nBasis)) + ones(fLen,nBasis)) .* ...
        (psdMix * ones(1,nBasis));
    initH = 0.5 * (abs(randn(nBasis,tLen)) + ones(nBasis,tLen));
    initQ = abs(initA).^2;
else
    initW = init.initW;
    initH = init.initH;
    initQ = init.initQ;
end

if nargin < 15
    beta = 0;
elseif isempty(beta)
    beta = 0;
end

% Multichannel Nonnegative Matrix Factorization
[Q, basisF, activationF, cost] = multinmf_inst_mu_beta(abs(xRaySpace).^2, ...
        beta, nIter, initQ, initW, initH, sourceNMFidx);

% Estimation of the source images in the ray space.
freqQ = permute(repmat(Q, 1,1,fLen), [3,1,2]);
estimateImage = multinmf_recons_im(xRaySpace, freqQ, basisF, ...
    activationF, sourceNMFidx);
