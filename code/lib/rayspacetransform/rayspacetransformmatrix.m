function [PSI] = rayspacetransformmatrix(f,c,d,L,mbar,W,qbar,sigma)
%%rayspacetransformmatrix
% This function computes the ray space transform matrix for multiple
% frequencies.
%
% Params:
%   - f: Frequency axis in Hz.
%   - c: Speed of sound.
%   - d: Distance between adjacent microphones.
%   - L: Number of microphones.
%   - mbar: m axis sampling interval.
%   - W: Length of m axis.
%   - qbar:	Q axis sampling interval.
%   - sigma: Gaussian window standard deviation.
%
% Returns:
%   - PSI: the ray space transform tensor for multiple frequencies.
%
% Copyright 2020 Mirco Pezzoli
% (mirco.pezzoli -at- polimi.it)
%
% This software is distributed under the terms of the GNU Public License
% version 3 (http://www.gnu.org/licenses/gpl.txt)
%
% Based on:
% L. Bianchi, F. Antonacci, A. Sarti and S. Tubaro,
% "The Ray Space Transform: A New Framework for Wave Field Processing,"
% in IEEE Transactions on Signal Processing, vol. 64, no. 21,
% pp. 5696-5706, 1 Nov.1, 2016.
% doi: 10.1109/TSP.2016.2591500
% URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7513427&isnumber=7562579
%

z = (0:d:d*(L-1))';                         % [L,1] microphone positions
m = ((0:mbar:(W-1)*mbar)-((W-1)/2*mbar))';  % [W,1] m axis
q = (0:qbar:z(end))';                       % [I,1] q axis
fLen = length(f);

[M, Q] = meshgrid(m,q);
[MM,MZ] = meshgrid(M(:),z);
[QQ,QZ] = meshgrid(Q(:),z);

PSI = zeros(L, size(MM,2), fLen);
% RST transformation matrix
for ff = 1:fLen
    PSI(:,:,ff) = d*exp(1i*(2*pi*f(ff)/c)*MZ.*MM./sqrt(1+MM.^2)) ...
        .* exp(-pi*(QZ-QQ).^2/sigma^2);
end

end

