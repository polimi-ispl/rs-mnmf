function y = spatialfilter(S, h, inverse)
%SPATIALFILTER This function efficiently filter the multichannel STFT S
% using the filter vector h
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
