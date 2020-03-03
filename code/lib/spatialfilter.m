function y = spatialfilter(S, h, inverse)
%% spatialfilter
% This function efficiently filter the multichannel STFT using the filter 
% vector h.
%
% Params:
%   - S: The multichannel input STFT of size freqN x timeN x micN
%   - h: The spatial filter given by the functions rayspacetransformmatrix
%   or beamspacetransformmatrix
%   - inverse: Flag true if we want to compute the spatial filter inverse 
%
% Copyright 2020 Mirco Pezzoli
% (mirco.pezzoli -at- polimi.it)
%
% This software is distributed under the terms of the GNU Public License
% version 3 (http://www.gnu.org/licenses/gpl.txt)
%
% If you use this code please cite this paper
%

    nDir = size(h,2);           % Number of looking directions
    fLen = size(S,1);           % Number of frequencies
    tLen = size(S,2);           % Number of time instants
    y = zeros(fLen,tLen,nDir);  % Filter output
    if inverse ~= true
        h = conj(h);
    end
    for aa = 1:nDir
        steeringV = h(:,aa,:);
        steeringV = permute(steeringV, [3,2,1]);
        steeringV = repmat(steeringV, 1,tLen,1);
        y(:,:,aa) = squeeze(sum(steeringV.*S,3));
    end
end
