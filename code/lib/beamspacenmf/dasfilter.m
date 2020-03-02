function h = dasfilter(fAx,thetaAx,d,nMic,c)
%DASFILTER This function computes the DAS filter 
%   Detailed explanation goes here
fLen = length(fAx);
nDoa = length(thetaAx);
h = zeros(nMic, nDoa, fLen);     % Steering Vector

for mm = 1:nMic
    for aa = 1:nDoa
        h(mm,aa,:) = exp(-1i*(2*pi*fAx/c) * (d*sin(thetaAx(aa))*(mm-1)));
    end
end
h = h ./ nMic;
for ff = 1:fLen
    sqrH = sqrtm(h(:,:,ff)'*h(:,:,ff));
    h(:,:,ff) = h(:,:,ff) *pinv(sqrH);
end

end

