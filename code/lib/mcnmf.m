function [estimateImage, Q, basisF, activationF] = mcnmf(micSTFT, fAx, nMic, sourceN, nBasisSource, nIter, conv, init)
%% mcnmf
% This function performs the Multichannel Nonnegative Matrix Factorization
% both in the instantaneous and convolutive mixture case.
% 
% Params:
%   - micSTFT: The multichannel STFTs of size freqN x timeN x micN.
%   - fAx: Frequency axis.
%   - nMic: Number of microphones.                   
%   - sourceN: Number of sources.
%   - nBasisSource: Number of basis function used  in the NMF for each 
%   source.
%   - nIter: Number of MCNMF iterations.
%   - conv: Flag to select convolutive (true) or instantaneous (false)
%   mixing model.
%   - init: Struct that contains the initialization of the algorithm if
%   empty the initialization is computed.
%   
% Returns:
%   - estimateImage: The estimated source images at the microphones.
%   - Q: The mixing coefficients.
%   - basisF: The basis functions of the sources.
%   - activationF: The activation functions of the sources.
%
% Copyright 2020 Mirco Pezzoli
% (mirco.pezzoli -at- polimi.it)
%
% This software is distributed under the terms of the GNU Public License
% version 3 (http://www.gnu.org/licenses/gpl.txt)
%
% Based on:
% A. Ozerov and C. Fevotte,
% "Multichannel nonnegative matrix factorization in convolutive mixtures for audio source separation,"
% IEEE Trans. on Audio, Speech and Lang. Proc. special issue on Signal Models and Representations
% of Musical and Environmental Sounds, vol. 18, no. 3, pp. 550-563, March 2010.
% Available: http://www.irisa.fr/metiss/ozerov/Publications/OzerovFevotte_IEEE_TASLP10.pdf


fLen = length(fAx);
tLen = size(micSTFT, 2);
nBasis = nBasisSource * sourceN;
sourceNMFidx = cell(1,sourceN);
for iSrc = 1:sourceN
    sourceNMFidx{iSrc} = (1:nBasisSource) + (iSrc-1)*nBasisSource;
end

%% Multichannel NMF
if conv == false
    fprintf('Multichannel NMF instantaneous mix...\n');
else
    fprintf('Multichannel NMF convolutive mix...\n');
end
if isempty(init)
    psdMix = 0.5 * mean(abs(micSTFT(:,:,1)).^2 + ....
        abs(micSTFT(:,:,2)).^2, 2);
    initW = 0.5 * (abs(randn(fLen,nBasis)) + ones(fLen,nBasis)) .* ...
    (psdMix * ones(1,nBasis));
    initH = 0.5 * (abs(randn(nBasis,tLen)) + ones(nBasis,tLen));
else
    initW = init.initW;
    initH = init.initH;
end

if conv == false
    if isempty(init)
        initA = 0.5 * (1.9 * abs(randn(nMic, sourceN)) + ...
            0.1 * ones(nMic, sourceN));
        initQ = abs(initA).^2;
    else
        initQ = init.initQ;
    end
    [Q, basisF, activationF, ~] = ...
        multinmf_inst_mu(abs(micSTFT).^2, nIter, initQ, initW, initH, ...
        sourceNMFidx);
    freqQ = permute(repmat(Q, 1,1,fLen), [3,1,2]);
    estimateImage = multinmf_recons_im(micSTFT, freqQ, basisF, ...
        activationF, sourceNMFidx);
else
    if isempty(init)
        initA = 0.5 * (1.9 * abs(randn(fLen, nMic, sourceN)) + ...
            0.1 * ones(fLen, nMic, sourceN));
        initQ = abs(initA).^2;
    else
        initQ = init.initQ;
    end

    [Q, basisF, activationF, ~] = ...
        multinmf_conv_mu(abs(micSTFT).^2, nIter, initQ, initW, initH, ...
        sourceNMFidx);
    estimateImage = multinmf_recons_im(micSTFT, Q,  basisF, ...
        activationF, sourceNMFidx);
end

