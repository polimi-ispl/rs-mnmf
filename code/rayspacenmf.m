function [estimateImage, Q, basisF, activationF, xRaySpace1, Wb, invWb, initQ] = rayspacenmf(micSTFT, mbar, Ws, qbar, sigma, fAx, d, nMic, c, sourceN, nBasisSource,...
    nIter, regParamters, init)

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
Wb = rayspacetransformmatrix(fAx,c,d,nMic,mbar,Ws,qbar,sigma);
I = size(Wb, 1);

invWb = zeros(size(Wb,2),size(Wb,1),fLen);
for ff = 2:fLen
    %    invWb(:,:,ff) = pinv(Wb(:,:,ff));
    [invWb(:,:,ff), tik(ff)] = svd_inverse_matrix(Wb(:,:,ff), regParamters);
    condInv(ff) = cond(invWb(:,:,ff));
end
% Rayspace signals
xRaySpace1 = spatialfilter(micSTFT, Wb,false);

%% MNMF of rayspace data
fprintf('MNMF on rayspace...\n');
if isempty(init)
    % Parameters initialization
    psdMix = 0.5 * (mean(abs(xRaySpace1(:,:,1)).^2 + abs(xRaySpace1(:,:,2)).^2, 2));
    % psdMix = mean(mean(abs(xRaySpace).^2, 3) + mean(abs(xRaySpace).^2, 3), 2);
    initA = 0.5 * (1.9 * abs(randn(I, sourceN)) + ...
        0.1 * ones(I, sourceN));
    % W is intialized so that its enegy follows mixture PSD
    initW = 0.5 * (abs(randn(fLen,nBasis)) + ones(fLen,nBasis)) .* ...
        (psdMix * ones(1,nBasis));
    initH = 0.5 * (abs(randn(nBasis,tLen)) + ones(nBasis,tLen));
    % run multichannel NMF MU rules
    initQ = abs(initA).^2;
    % mockUp = zeros(I,sourceN);
    % for ss = 1:sourceN
    %     mockUp(:,ss) = squeeze(mean(mean(abs(raySpaceRef{ss}).^2,3),2));
    % end
    % initQ = initQ*10e2;
else
    initW = init.initW;
    initH = init.initH;
    initA = init.initA;
    initQ = init.initQ;
end

[Q, basisF, activationF, cost] = ...
    multinmf_inst_mu(abs(xRaySpace1).^2, nIter, initQ, initW, initH,...
    sourceNMFidx);

freqQ = permute(repmat(Q, 1,1,fLen), [3,1,2]);
estimateImage = multinmf_recons_im(xRaySpace1, freqQ, basisF, ...
    activationF, sourceNMFidx);
