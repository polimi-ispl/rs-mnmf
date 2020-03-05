%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% RS-MCNMF example                                                         
% Ray-Space-Based Multichannel Nonnegative Matrix Factorization for Audio
% Source Separation.
% This script shows an example of audio source separation using the
% RS-MCNMF algorithm compared to the BS-MCNMF and the classic MCNMF.
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
fs = 8000;      % Sampling frequency [Hz]
c = 340;        % Speed of sound [m/s]
sourceN = 2;    % Number of sources

% STFT parameters
winLength = 256;        % Window length in samples
analysisWin = hamming(winLength,'periodic');
synthesisWin = hamming(winLength,'periodic');
hop = winLength / 4;    % Hop size (75% overlap)
nfft = 2*2^nextpow2(winLength);

% ULA parameters
d = 0.03;           % Distance between two adjacent microphones
nMic = 32;          % Number of microphones
y = (0:d:d*(nMic-1))';                 % mic y coordinate
micPos = [zeros(nMic,1) y];            % mic position in 2D

% Ray Space Transform parameters
mubar = 0.06;       % m axis sampling interval
Ws = 4;             % length of m axis
nubar = 4*d;        % nu axis sampling interval
sigma = 8*d;        % gaussian window standard deviation
mu = ((0:mubar:(Ws-1)*mubar)-((Ws-1)/2*mubar))';       % [W,1] mu axis
nu = (0:nubar:micPos(end,end))';                       % [L,1] nu axis
t = 1:nMic;
tik = 0.5;          % Regularization parameter for the psiTilde

% Beam Space paramteres
M = nMic;           % Number of plane waves 

% MCNMF parameters
nBasisSource = 12;
nBasis = nBasisSource * sourceN;
sourceNMFidx = cell(1,sourceN);
for iSrc = 1:sourceN
    sourceNMFidx{iSrc} = (1:nBasisSource) + (iSrc-1)*nBasisSource;
end
nIter = 20;
mcnmfConv = true;

% Directory where the impulse resposes are stored
dataPath = ['..' filesep 'data' filesep 'example'];

%% Geometric setup
% Source location from office room experiment
load([dataPath filesep 'sourcePosition.mat']);

figure(1)
scatter(micPos(:,1), micPos(:,2))
hold on
scatter(sourcePosition(:,1), sourcePosition(:,2))
title('Setup'), xlim([-0.1, 3.8]), grid on
legend('Mics', 'Sources')

%% Microphone array signals
fprintf('Load the signal...\n');
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

% Ray-Space-Transformed reference signals
raySpaceRef = cell(sourceN,1);  
% Source signal estimate based on basis and activation functions
sourceRecSTFT = cell(sourceN,1);
sourceRec = cell(1, sourceN);
% Ray Space source images estimated by the RS-MCNMF
raySpaceEstimateImage = cell(sourceN,1);

% Initialization of the RS-MCNMF algorithm
psdMix = 0.5 * (mean(abs(micSTFT(:,:,1)).^2 + abs(micSTFT(:,:,2)).^2, 2));
init.initA = 0.5 * (1.9 * abs(randn(nMic, sourceN)) + ...
    0.1 * ones(nMic, sourceN));
% W is intialized so that its enegy follows mixture PSD
init.initW = 0.5 * (abs(randn(fLen,nBasis)) + ones(fLen,nBasis)) .* ...
    (psdMix * ones(1,nBasis));
init.initH = 0.5 * (abs(randn(nBasis,tLen)) + ones(nBasis,tLen));
init.initQ = abs(init.initA).^2;

% Source separation using MCNMF in the ray space
[estimateImage, Q, basisF, activationF, xRaySpace, psi, ...
    invPsi, initQ] = rayspacenmf(micSTFT, mubar, Ws, nubar, sigma, fAx, ...
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

% Metrics for the estimation of the sources only
sourceEstimation = cell2mat(sourceRec.');
[sourceRaySDR, sourceRaySIR, sourceRaySAR, perm] = ...
    bss_eval_sources(sourceEstimation, sourceSignal.');

estimateImage = estimateImage(:,:,perm,:);
sourceRecSTFT = sourceRecSTFT(perm);
estimateImage = estimateImage(:,:,perm,:);
for ss = 1:sourceN
    raySpaceEstimateImage{ss} = squeeze(estimateImage(:,:,ss,:));
    raySpaceEstimateImage{ss} = permute(raySpaceEstimateImage{ss}, ...
        [3,2,1]);
end

%% Plot the Ray space data
fprintf('Show Ray Space separation results...\n');
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

%% Estimated source image at the microphone 
fprintf('RS-MCNMF estimated source image at the microphones...\n');
istftParams.analysisWin = analysisWin;
istftParams.synthesisWin = synthesisWin;
istftParams.hop = hop;
istftParams.nfft = nfft;

% iRST of the estimated images
rsmcnmfEstimate = arraysignalreconstruction(estimateImage, ...
    referenceSignal, invPsi, sourceN, nMic, istftParams, fs);

% Dummy variables for bss_eval
% Matrix of the reference signals
ref = cat(3, referenceSignal{:});      
% Matrix of the estimates given by the RS-MCNMF
est = cat(3, rsmcnmfEstimate{:});     

raySDR = zeros(nMic,sourceN);
raySIR = zeros(nMic,sourceN);
raySAR = zeros(nMic,sourceN);

for mm = 1:nMic
    [raySDR(mm,:), raySIR(mm,:), raySAR(mm,:)] = bss_eval_sources( ...
        squeeze(est(:,mm,:)).', squeeze(ref(:,mm,:)).');
end

rsmcnmfSDR = mean(raySDR);
rsmcnmfSIR = mean(raySIR);
rsmcnmfSAR = mean(raySAR);

%% Source separation through BS-MCNMF

% Beam-Space-Transformed reference signals
beamSpaceRef = cell(sourceN,1);  
% Source signal estimate based on basis and activation functions
sourceRecSTFTBeam = cell(sourceN,1);
sourceRecBeam = cell(1,sourceN);
% Beam Space source images estimated by the BS-MCNMF
beamSpaceEstimateImage = cell(sourceN,1);

% Source separation using the BS-MCNMF
[estimateImageBeam, Q, basisF, activationF, xBeamSpace, invWbB] = ...
    beamspacenmf(micSTFT, M, fAx, d, nMic, c, sourceN, nBasisSource, ...
    nIter, init);

% Beam space reference and estimate
for ss = 1:sourceN
    beamSpaceRef{ss} = spatialfilter(referenceSTFT{ss},psi,false);
    beamSpaceRef{ss} = permute(beamSpaceRef{ss}, [3,2,1]);
    sourceRecSTFTBeam{ss} = sqrt(basisF(:,sourceNMFidx{ss}) * ...
        activationF(sourceNMFidx{ss} ,:)) .* angle(micSTFT(:,:,1));
    sourceRecBeam{ss} = istft(sourceRecSTFTBeam{ss}, ...
        analysisWin, synthesisWin, hop, nfft, fs);
end

sourceEstimationBeam = cell2mat(sourceRecBeam.');
% Metrics for the estimation of the sources only
[sourceBeamSDR, sourceBeamSIR, sourceBeamSAR, perm] = ...
    bss_eval_sources(sourceEstimationBeam, sourceSignal.');

estimateImageBeam = estimateImageBeam(:,:,perm,:);
sourceRecSTFTBeam = sourceRecSTFTBeam(perm);
sourceRecBeam = sourceRecBeam(perm);
for ss = 1:sourceN
    beamSpaceEstimateImage{ss} = squeeze(estimateImageBeam(:,:,ss,:));
    beamSpaceEstimateImage{ss} = permute(beamSpaceEstimateImage{ss}, ...
        [3,2,1]);
end

%% Plot the Beam Space data
fprintf('Show Beam Space separation results...\n');
fIdx = 35;      % Frequency index to show
% Beam Space mixture
mixBeam = squeeze(abs(xBeamSpace(fIdx ,:,:))).'.^2;
% Beam Space reference 1
refBeam1 = squeeze(abs(beamSpaceRef{1}(:,:,fIdx))).^2;
% Beam Space reference 2
refBeam2 = squeeze(abs(beamSpaceRef{2}(:,:,fIdx))).^2;
% Estimated Beam Space 1
estBeam1 = squeeze(abs(beamSpaceEstimateImage{1}(:,:,fIdx))).^2;
% Estimated Beam Space 2
estBeam2 = squeeze(abs(beamSpaceEstimateImage{2}(:,:,fIdx))).^2;

maxC = max([mixBeam(:); refBeam1(:); refBeam2(:); estBeam1(:); ...
    estBeam2(:)]);
minC = min([mixBeam(:); refBeam1(:); refBeam2(:); estBeam1(:); ...
    estBeam2(:)]);

figure(4)
imagesc(tAx, t, db(mixBeam), db([minC, maxC]))
xlabel('Time [s]'), ylabel('Beam space point m');
colorbar
title(['Beam Space Mixture at ' num2str(fAx(fIdx)), ' [Hz]']);

figure(5)
subplot(2,2,1)
imagesc(tAx, t, db(refBeam1), db([minC, maxC]))
xlabel('Time [s]'), ylabel('Beam space point m');
colorbar
title('Beam Space of source 1 image');

subplot(2,2,2)
imagesc(tAx, t, db(refBeam2), db([minC, maxC]))
xlabel('Time [s]'), ylabel('Beam space point m');
colorbar
title('Beam Space of source 2 image');

subplot(2,2,3)
imagesc(tAx, t, db(estBeam1), db([minC, maxC]))
xlabel('Time [s]'), ylabel('Beam space point m');
colorbar
title('Estimated Beam Space source 1 image');

subplot(2,2,4)
imagesc(tAx, t, db(estBeam2), db([minC, maxC]))
xlabel('Time [s]'), ylabel('Beam space point m');
colorbar
title('Estimated Beam Space source 2 image');

%% Estimated source image at the microphone 
fprintf('BS-MCNMF estimated source image at the microphones...\n');
istftParams.analysisWin = analysisWin;
istftParams.synthesisWin = synthesisWin;
istftParams.hop = hop;
istftParams.nfft = nfft;

bsmcnmfEstimate = arraysignalreconstruction(estimateImageBeam, ...
    referenceSignal, invWbB, sourceN, nMic, istftParams, fs);

% Dummy variables for bss_eval
% Matrix of the reference signals
ref = cat(3, referenceSignal{:});      
% Matrix of the estimates given by the BS-MCNMF
est = cat(3, bsmcnmfEstimate{:});     
beamSDR = zeros(nMic,sourceN);
beamSIR = zeros(nMic,sourceN);
beamSAR = zeros(nMic,sourceN);

for mm = 1:nMic
    [beamSDR(mm,:), beamSIR(mm,:), beamSAR(mm,:)] = bss_eval_sources( ...
        squeeze(est(:,mm,:)).', squeeze(ref(:,mm,:)).');
end

bsmcnmfSDR = mean(beamSDR);
bsmcnmfSIR = mean(beamSIR);
bsmcnmfSAR = mean(beamSAR);

%% Source separation through MCNMF
sourceRecSTFTMulti = cell(sourceN,1);
sourceRecMulti = cell(1,sourceN);
% Source separation with MCNMF convolutive mixture model
initConv = init;
initConv.initQ = permute(repmat(init.initQ, 1, 1, fLen), [3,1,2]);
[estimateImageMulti, Q, basisF, activationF] = ...
    mcnmf(micSTFT, fAx, nMic, sourceN, nBasisSource,...
    nIter, mcnmfConv, initConv);

% Multichannel source signal estimate
for ss = 1:sourceN
    sourceRecSTFTMulti{ss} = sqrt(basisF(:,sourceNMFidx{ss}) * ...
        activationF(sourceNMFidx{ss} ,:)) .* angle(micSTFT(:,:,1));
    sourceRecMulti{ss} = istft(sourceRecSTFTMulti{ss}, ...
        analysisWin, synthesisWin, hop, nfft, fs);
end


sourceEstimationMulti = cell2mat(sourceRecMulti.');
% Metrics for the estimation of the sources only
[sourceMultiSDR, sourceMultiSIR, sourceMultiSAR, perm] = ...
    bss_eval_sources(sourceEstimationMulti, sourceSignal.');

estimateImageMulti = estimateImageMulti(:,:,perm,:);
sourceRecSTFTMulti = sourceRecSTFTMulti(perm);
sourceRecMulti = sourceRecMulti(perm);

%% Estimated source image at the microphone 
fprintf('MCNMF estimated source image at the microphones...\n');

signalLenSamp = length(referenceSignal{1}(:,1));
mcnmfEstimate = cell(sourceN, 1);

% Compute the time domain signal at the microphones
for ss = 1:sourceN
    sTilde = squeeze(estimateImageMulti(:,:,ss,:));
    for mm = 1:nMic
        aux = istft(sTilde(:,:,mm), ...
            analysisWin, synthesisWin, hop, nfft, fs);
        [auxAligned, referenceAligned] = ...
            alignsignals(aux, referenceSignal{ss}(:,mm));
        
        mcnmfEstimate{ss}(:,mm) = auxAligned(1:signalLenSamp).';
    end
end

% Dummy variable for bss_eval
ref = cat(3, referenceSignal{:});
est = cat(3, mcnmfEstimate{:});
multiSDR = zeros(nMic, sourceN);
multiSIR = zeros(nMic, sourceN);
multiSAR = zeros(nMic, sourceN);

for mm = 1:nMic
    [multiSDR(mm,:), multiSIR(mm,:), multiSAR(mm,:)] = bss_eval_sources(...
        squeeze(est(:,mm,:)).', squeeze(ref(:,mm,:)).');
end

mcnmfSDR = mean(multiSDR);
mcnmfSIR = mean(multiSIR);
mcnmfSAR = mean(multiSAR);

%% Show metrics

figure(6)
categ = categorical({'SAR', 'SIR', 'SDR'});
plotBar = [bsmcnmfSAR(1), mcnmfSAR(1), rsmcnmfSAR(1);...
bsmcnmfSIR(1), mcnmfSIR(1), rsmcnmfSIR(1);...
    bsmcnmfSDR(1), mcnmfSDR(1), rsmcnmfSDR(1)];
subplot(2,1,1)
bar(categ, plotBar)
plotBar = [bsmcnmfSAR(2), mcnmfSAR(2), rsmcnmfSAR(2); ...
    bsmcnmfSIR(2), mcnmfSIR(2), rsmcnmfSIR(2);...
    bsmcnmfSDR(2), mcnmfSDR(2), rsmcnmfSDR(2)];
title('Source 1 avg metrics')
legend({'Beam Space', 'Multi channel', 'Ray Space'}, 'Location', 'best');
subplot(2,1,2)
bar(categ, plotBar)
title('Source 2 avg metrics')
legend({'Beam Space', 'Multi Channel', 'Ray Space'}, 'Location', 'best');

%% Listening comparison
fprintf('Listening comparison..\n');
micIdx = 1;

% First Source
fprintf('\nSource 1 (female voice)\n')
fprintf('Press any key to start\n');
pause();
fprintf('Reference signal...\n');
soundsc(referenceSignal{1}(:,micIdx), fs)
pause(signalLenSamp/fs);


fprintf('RS-MCNMF Estimation... Press any key to start\n');
pause();
soundsc(rsmcnmfEstimate{1}(:,micIdx), fs)
pause(signalLenSamp/fs);


fprintf('MCNMF Estimation... Press any key to start\n');
pause();
soundsc(mcnmfEstimate{1}(:,micIdx), fs)
pause(signalLenSamp/fs);

fprintf('BS-MCNMF Estimation... Press any key to start\n'); 
pause();
soundsc(bsmcnmfEstimate{1}(:,micIdx), fs)
pause(signalLenSamp/fs);

% Second soure
fprintf('\nSource 2 (male voice)\n')

fprintf('Press any key to start\n');
pause();
fprintf('Reference signal...\n');  
soundsc(referenceSignal{2}(:,micIdx), fs)
pause(signalLenSamp/fs);

fprintf('Press any key to start\n');
fprintf('RS-MCNMF Estimation... Press any key to start\n');
pause();
soundsc(rsmcnmfEstimate{2}(:,micIdx), fs)
pause(signalLenSamp/fs);

fprintf('MCNMF Estimation... Press any key to start\n');
pause();
soundsc(mcnmfEstimate{2}(:,micIdx), fs)
pause(signalLenSamp/fs);

fprintf('BS-MCNMF Estimation... Press any key to start\n');
pause();
soundsc(bsmcnmfEstimate{2}(:,micIdx), fs)
pause(signalLenSamp/fs);














