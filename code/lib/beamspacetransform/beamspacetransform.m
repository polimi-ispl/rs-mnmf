function h = beamspacetransform(fAx,thetaAx,d,nMic,c)
%% beamspacetransform 
% This function computes the DAS stering vectors used for the Beam Space
% transform and the Beam Space transform matrix
% Params: 
%   - fAx: The temporal frequency axis
%   - thetaAx: The direction axis (DOA) 
%   - d: distance between two consecutive microphones                      
%   - nMic: number of microphones
%   - c: sound speed in air
% Returns:
%   - h: the beamspace transform matrix
%
% Copyright 2020 Mirco Pezzoli
% (mirco.pezzoli -at- polimi.it)
%
% This software is distributed under the terms of the GNU Public License
% version 3 (http://www.gnu.org/licenses/gpl.txt)
%

fLen = length(fAx);                 % Number of frequency bins
nDoa = length(thetaAx);             % Number of directions
h = zeros(nMic, nDoa, fLen);        % Beam Space transform matrix

for mm = 1:nMic
    for aa = 1:nDoa
        h(mm,aa,:) = exp(-1i*(2*pi*fAx/c) * (d*sin(thetaAx(aa))*(mm-1)));
    end
end
h = h ./ nMic;
for ff = 1:fLen
    sqrH = sqrtm(h(:,:,ff)' * h(:,:,ff));
    h(:,:,ff) = h(:,:,ff) * pinv(sqrH);
end

end

