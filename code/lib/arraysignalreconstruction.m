function [beamSignalEstimate, nmse] = arraysignalreconstruction(estimateImage, referenceSignal, invWb, sourceN, nMic, istftParams, fs) 

fprintf('Signal estimation...\n');
sTilde = cell(sourceN,1);
nmse = sTilde;

beamSignalEstimate = cell(sourceN,1);
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
        
        beamSignalEstimate{ss}(:,mm) = auxAligned(1:signalLen).';
        
        nmse{ss}(mm) = NMSE(beamSignalEstimate{ss}(:,mm) ./ ...
            max(beamSignalEstimate{ss}(:,mm)), ...
            referenceAligned(1:signalLen) ./...
            max(referenceAligned(1:signalLen)));
    end
end
