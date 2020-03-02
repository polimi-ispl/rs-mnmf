function [estimateImage, Q, basisF, activationF, xTilde, invWbB] = beamspacenmf(micSTFT, I, fAx, d, nMic, c, sourceN, nBasisSource, nIter, init)
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


%% Beamspace projection
fprintf('Beamspace transform...\n');
thetaAx = linspace(-pi/2, pi/2, I);
% Beamspace transformation matrix
Wb = beamspacetransform(fAx, thetaAx, d, nMic, c);
invWbB = permute(Wb, [2,1,3]);

% Beamspace signals
xTilde = spatialfilter(micSTFT, Wb,false);

%% MNMF of beamspace data
fprintf('MNMF on beamspace...\n');
% Parameters initialization
if isempty(init)
    psdMix = (mean(abs(xTilde(:,:,1)).^2 + abs(xTilde(:,:,nMic)).^2, 2));
    initA = 0.5 * (1.9 * abs(randn(I, sourceN)) + ...
        0.1 * ones(I, sourceN));
    % initA = 0.5 * ones(W, sourceN);
    % W is intialized so that its enegy follows mixture PSD
    initW = 0.5 * (abs(randn(fLen,nBasis)) + ones(fLen,nBasis)) .* ...
        (psdMix * ones(1,nBasis));
    % initW = (psdMix * ones(1,nBasis));
    initH = 0.5 * (abs(randn(nBasis,tLen)) + ones(nBasis,tLen));
    % initH = 0.5 * (abs(randn(nBasis,tLen)) + ones(nBasis,tLen));
    % run 500 iterations of multichannel NMF MU rules
    initQ = abs(initA).^2;
else
    initW = init.initW;
    initH = init.initH;
    initA = init.initA;
    initQ = init.initQ;
end

[Q, basisF, activationF, ~] = ...
    multinmf_inst_mu(abs(xTilde).^2, nIter, initQ, initW, initH,...
    sourceNMFidx);

freqQ = permute(repmat(Q, 1,1,fLen), [3,1,2]);
estimateImage = multinmf_recons_im(xTilde, freqQ, basisF, ...
    activationF, sourceNMFidx);

