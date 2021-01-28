function [estimatedSignal, nmse] = arraysignalreconstruction(estimateImage, referenceSignal, invWb, sourceN, nMic, istftParams, fs) 
%% arraysignalreconstruction
% This function recontructs the microphone signals in time domain from
% their representation in the ray space domain. 
% 
% Params:
%   - estimateImage: The ray space or beam space data given by the source
%       image estimation (MNMF separation)
%   - referenceSignal: The reference source signals at the microphones.
%   - invWb: The inverse transform matrix
%   - sourceN: The number of sources                                       
%   - nMic: the number of microphones
%   - istftParams: struct containing the istft parameters analysisWin,
%       synthesisWin, hop and nfft
%   - fs: sampling frequency
%
% Returns:
%   - estimatedSignal: 1xsourceN cell array of the microphone signals
%   - nmse: normalized mean squared error of the estimation with respect to
%   the reference signals
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


fprintf('Estimation of the source images...\n');
sTilde = cell(sourceN,1);
nmse = sTilde;

estimatedSignal = cell(sourceN,1);
signalLen = length(referenceSignal{1}(:,1));
analysisWin = istftParams.analysisWin;
synthesisWin = istftParams.synthesisWin;
hop = istftParams.hop;
nfft = istftParams.nfft;

for ss = 1:sourceN
    aux = squeeze(estimateImage(:,:,ss,:));
    sTilde{ss} = spatialfilter(aux, invWb, true);
    for mm = 1:nMic
        aux = istft(sTilde{ss}(:,:,mm), ...
            analysisWin, synthesisWin, hop, nfft, fs);
        [auxAligned, referenceAligned] = ...
            alignsignals(aux, referenceSignal{ss}(:,mm));
        if length(auxAligned) < signalLen
            diff = signalLen - length(auxAligned);
            auxAligned = [auxAligned, zeros(1, diff)];
        end
        estimatedSignal{ss}(:,mm) = auxAligned(1:signalLen).';
        
        nmse{ss}(mm) = NMSE(estimatedSignal{ss}(:,mm) ./ ...
            max(estimatedSignal{ss}(:,mm)), ...
            referenceAligned(1:signalLen) ./...
            max(referenceAligned(1:signalLen)));
    end
end
