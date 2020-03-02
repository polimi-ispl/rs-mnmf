function [estimateImage, Q, basisF, activationF] = mcnmf(micSTFT, fAx, nMic, sourceN, nBasisSource, nIter, conv, init)
% Copyright 2020 Mirco Pezzoli
% (mirco.pezzoli -at- polimi.it)
%
% This software is distributed under the terms of the GNU Public License
% version 3 (http://www.gnu.org/licenses/gpl.txt)
%
% If you use this code please cite this paper
%

fLen = length(fAx);
tLen = size(micSTFT, 2);
nBasis = nBasisSource * sourceN;
sourceNMFidx = cell(1,sourceN);
for iSrc = 1:sourceN
    sourceNMFidx{iSrc} = (1:nBasisSource) + (iSrc-1)*nBasisSource;
end

%% Multichannel NMF
if conv == false
    fprintf('Multichannel NMF instant. mix...\n');
else
    fprintf('Multichannel NMF convolutive mix...\n');
end
if isempty(init)
    psdMix = 0.5 * mean(abs(micSTFT(:,:,1)).^2 + abs(micSTFT(:,:,2)).^2, 2);
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
        initA = init.initA;
        initQ = init.initQ;
    end
    [Q, basisF, activationF, ~] = ...
        multinmf_inst_mu(abs(micSTFT).^2, nIter, initQ, initW, initH,...
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
        initA = init.initA;
        initQ = init.initQ;
    end

    [Q, basisF, activationF, ~] = ...
        multinmf_conv_mu(abs(micSTFT).^2, nIter, initQ, initW, initH,...
        sourceNMFidx);
    estimateImage = multinmf_recons_im(micSTFT, Q,  basisF, ...
        activationF, sourceNMFidx);
end
% %% MNMF of beamspace data
% fprintf('MNMF on beamspace...\n');
% % Parameters initialization
% psdMix = (mean(abs(xTilde(:,:,1)).^2 + abs(xTilde(:,:,nMic)).^2, 2));
% initA = 0.5 * (1.5 * abs(randn(I, sourceN)) + ...
%     0.1 * ones(I, sourceN));
% % initA = 0.5 * ones(W, sourceN);
% % W is intialized so that its enegy follows mixture PSD
% initW = 0.5 * (abs(randn(fLen,nBasis)) + ones(fLen,nBasis)) .* ...
%     (psdMix * ones(1,nBasis));
% % initW = (psdMix * ones(1,nBasis));
% initH = 0.5 * (abs(randn(nBasis,tLen)) + ones(nBasis,tLen));
% % initH = 0.5 * (abs(randn(nBasis,tLen)) + ones(nBasis,tLen));
% % run 500 iterations of multichannel NMF MU rules
% initQ = abs(initA).^2;
%
% [Q, basisF, activationF, ~] = ...
%     multinmf_inst_mu(abs(xTilde).^2, nIter, initQ, initW, initH,...
%     sourceNMFidx);
%
% freqQ = permute(repmat(Q, 1,1,fLen), [3,1,2]);
% estimateImage = multinmf_recons_im(xTilde, freqQ, basisF, ...
%     activationF, sourceNMFidx);

