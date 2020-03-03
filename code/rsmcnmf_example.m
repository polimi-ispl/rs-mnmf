%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% RS-MCNMF example                                                         
% Ray-Space-Based Multichannel Nonnegative Matrix Factorization for Audio
% Source Separation.
% This script shows an example of audio source separation using the
% RS-MCNMF algorithm
%
% Copyright 2020 Mirco Pezzoli
% (mirco.pezzoli -at- polimi.it)
%
% This software is distributed under the terms of the GNU Public License
% version 3 (http://www.gnu.org/licenses/gpl.txt)
%
% If you use this code please cite this paper
%
clear
close all
clc
%% Setup
fprintf('Setup...\n');
addpath(genpath('..'))
% General parameters
fs = 8000;
c = 340;        % Speed of sound m/s
sourceN = 2;    % Number of sources

% STFT parameters
winLength = 256;
analysisWin = hamming(winLength,'periodic');
synthesisWin = hamming(winLength,'periodic');
hop = winLength / 4;
nfft = 2*2^nextpow2(winLength);

% ULA parameters
d = 0.03;           % Distance between two adjacent microphones
nMic = 32;          % Number of microphones
y = (0:d:d*(nMic-1))';                 % mic z coordinate
micPos = [zeros(nMic,1) y];            % mic position in 2D

% Ray Space Transform parameters
mbar = 0.06;        % m axis sampling interval
Ws = 4;             % length of m axis
qbar = 4*d;         % q axis sampling interval
sigma = 8*d;        % gaussian window standard deviation
m = ((0:mbar:(Ws-1)*mbar)-((Ws-1)/2*mbar))';  % [W,1] m axis
tik = 0.5;
q = (0:qbar:micPos(end,end))';                       % [I,1] q axis
t = 1:nMic;

% MNMF parameters
nBasisSource = 12;
nBasis = nBasisSource * sourceN;
sourceNMFidx = cell(1,sourceN);
for iSrc = 1:sourceN
    sourceNMFidx{iSrc} = (1:nBasisSource) + (iSrc-1)*nBasisSource;
end
nIter = 20;

% Directory where the impulse resposes are stored
dataPath = ['..' filesep 'data' filesep 'example'];

%% Geometric setup
% Source location from office room experiment source number 7 and 9
load([dataPath filesep 'sourcePosition.mat']);

figure(1)
scatter(micPos(:,1), micPos(:,2))
hold on
scatter(sourcePosition(:,1), sourcePosition(:,2))
title('Setup'), xlim([-0.1, 3.8]), grid on
legend('Mics', 'Sources')

%% Microphone array signals
sourceSignal = audioread([dataPath filesep 'sourceSignal.wav']);
% Load the mixtures
micSignal = audioread([dataPath filesep 'array_mixture.wav']);

% Load the reference signals for the first source
referenceSignal{1} = audioread([dataPath filesep 'array_reference_1.wav']);

% Load the reference signals for the second source
referenceSignal{2} = audioread([dataPath filesep 'array_reference_2.wav']);

referenceSTFT = cell(1,sourceN);
for mm = 1:nMic
     [micSTFT(:,:,mm), fAx, tAx] = stft(micSignal(:,mm), analysisWin, ...
         hop, nfft, fs);
end
for ss = 1:sourceN
    for mm = 1:nMic
        referenceSTFT{ss}(:,:,mm) = stft(referenceSignal{ss}(:,mm), ...
            analysisWin, hop, nfft, fs);
    end
end
tLen = length(tAx);         % Length of time axis
fLen = nfft/2+1;            % Length of frequency axis

%% Source separation through RS-MCNMF

init = [];
% Source separation using MCNMF in the ray space
[estimateImage, Q, basisF, activationF, xRaySpace, psi, ...
    invPsi, initQ] = rayspacenmf(micSTFT, mbar, Ws, qbar, sigma, fAx, ...
    d, nMic, c, sourceN, nBasisSource,...
    nIter, tik, init);

% Ray space reference and estimate
for ss = 1:sourceN
    raySpaceRef{ss} = spatialfilter(referenceSTFT{ss},psi,false);
    raySpaceRef{ss} = permute(raySpaceRef{ss}, [3,2,1]);
    sourceRecSTFT{ss} = sqrt(basisF(:,sourceNMFidx{ss}) * ...
        activationF(sourceNMFidx{ss} ,:)) .* angle(micSTFT(:,:,1));
    sourceRec{ss} = istft(sourceRecSTFT{ss}, ...
        analysisWin, synthesisWin, hop, nfft, fs);
end

sourceEstimation = cell2mat(sourceRec.');
[sourceRaySDR, sourceRaySIR, sourceRaySAR, perm] = bss_eval_sources(sourceEstimation, sourceSignal.');
estimateImage = estimateImage(:,:,perm,:);
sourceRecSTFT = sourceRecSTFT(perm);
estimateImage = estimateImage(:,:,perm,:);
for ss = 1:sourceN
    raySpaceEstimateImage{ss} = squeeze(estimateImage(:,:,ss,:));
    raySpaceEstimateImage{ss} = permute(raySpaceEstimateImage{ss}, [3,2,1]);
end
%% Plot the Ray space data
fIdx = 35;      % Frequency index to show
% Ray Space mixture
mixRay = squeeze(abs(xRaySpace(fIdx ,:,:))).'.^2;
% Ray Space reference 1
refRay1 = squeeze(abs(raySpaceRef{1}(:,:,fIdx))).^2;
% Ray Space reference 2
refRay2 = squeeze(abs(raySpaceRef{2}(:,:,fIdx))).^2;
% Estimated Ray Space 1
estRay1 = squeeze(abs(raySpaceEstimateImage{1}(:,:,fIdx))).^2;
% Estimated Ray Space 2
estRay2 = squeeze(abs(raySpaceEstimateImage{2}(:,:,fIdx))).^2;

maxC = max([mixRay(:); refRay1(:); refRay2(:); estRay1(:); estRay2(:)]);
minC = min([mixRay(:); refRay1(:); refRay2(:); estRay1(:); estRay2(:)]);
figure(2)
imagesc(tAx, t, mixRay, [minC, maxC])
xlabel('Time [s]'), ylabel('Ray space point t');
colorbar
title(['Ray Space Mixture at ' num2str(fAx(fIdx)), ' [Hz]']);

figure(3)
subplot(2,2,1)
imagesc(tAx, t, refRay1, [minC, maxC])
xlabel('Time [s]'), ylabel('Ray space point t');
colorbar
title('Ray Space of source 1 image');

subplot(2,2,2)
imagesc(tAx, t, refRay2, [minC, maxC])
xlabel('Time [s]'), ylabel('Ray space point t');
colorbar
title('Ray Space of source 2 image');

subplot(2,2,3)
imagesc(tAx, t, estRay1, [minC, maxC])
xlabel('Time [s]'), ylabel('Ray space point t');
colorbar
title('Estimated Ray Space source 1 image');

subplot(2,2,4)
imagesc(tAx, t, estRay2, [minC, maxC])
xlabel('Time [s]'), ylabel('Ray space point t');
colorbar
title('Estimated Ray Space source 2 image');
