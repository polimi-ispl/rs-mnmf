function [estimateImage, Q, basisF, activationF, xTilde, invWbB] = beamspacenmf(micSTFT, I, fAx, d, nMic, c, sourceN, nBasisSource, nIter, init)
% beamspacenmf
% This function performs the source separation using the Beam-Space-Based
% Multichannel Nonnegative  Matrix factorization.
% The beam space representation of the array signal is computed and the
% source images in the Beam Space are computed by the MCNMF. 
% Params:
%   - micSTFT: The multichannel STFTs of size freqN x timeN x micN.
%   - I: The number of Beam Space bin (plane wave directions).
%   - fAx: Frequency axis.
%   - d: Distance between two consecutive microphones.
%   - nMic: Number of microphones.                   
%   - c: Sound speed. 
%   - sourceN: Number of sources.
%   - nBasisSource: Number of basis function used  in the NMF for each 
%   source.
%   - nIter: Number of MCNMF iterations.
%   - init: Struct that contains the initialization of the algorithm if
%   empty the initialization is computed.
% Returns:
%   - estimateImage: The estimated source images in the Beam Space.
%   - Q: The Beam Space mixing coefficients.
%   - basisF: The basis functions of the sources.
%   - activationF: The activation functions of the sources.
%   - xTilde: The Beam Space data.
%   - invWbB: The inverse Beam Space transform matrix.
%
% Copyright 2020 Mirco Pezzoli
% (mirco.pezzoli -at- polimi.it)
%
% This software is distributed under the terms of the GNU Public License
% version 3 (http://www.gnu.org/licenses/gpl.txt)
%
% Based on:
% S. Lee, S. H. Park and K. Sung, 
% "Beamspace-Domain Multichannel Nonnegative Matrix Factorization for Audio Source Separation," 
% in IEEE Signal Processing Letters, vol. 19, no. 1, pp. 43-46, Jan. 2012.
% doi: 10.1109/LSP.2011.2173192
% URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6058587&isnumber=6071747


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

[Q, basisF, activationF, ~] = ...
    multinmf_inst_mu(abs(xTilde).^2, nIter, initQ, initW, initH,...
    sourceNMFidx);

freqQ = permute(repmat(Q, 1,1,fLen), [3,1,2]);
estimateImage = multinmf_recons_im(xTilde, freqQ, basisF, ...
    activationF, sourceNMFidx);

