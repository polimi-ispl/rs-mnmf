function [PSI] = rayspacetransformmatrix(f,c,d,L,mbar,W,qbar,sigma)
%rayspacetransformmatrix
%   This fucntion computes the ray space transform matrix for multiple
%   frequencies in f.
%   Args:
%       - f:	frequency axis in Hz
%       - c:	speed of sound
%       - d:	distance between adjacent microphones
%       - L:    Number of microphones
%       - mbar: m axis sampling interval
%       - W:    length of m axis
%       - qbar:	q axis sampling interval
%       - sigma:gaussian window standard deviation
% 
%   Returns:
%       - PSI: the ray space transform tensor for multiple frequencies
%
% Mirco Pezzoli 2019

z = (0:d:d*(L-1))';                         % [L,1] microphone positions
m = ((0:mbar:(W-1)*mbar)-((W-1)/2*mbar))';  % [W,1] m axis
q = (0:qbar:z(end))';                       % [I,1] q axis
I = length(q);                              % number of frames
fLen = length(f);

[M, Q] = meshgrid(m,q);
[MM,MZ] = meshgrid(M(:),z);
[QQ,QZ] = meshgrid(Q(:),z);

PSI = zeros(L, size(MM,2), fLen);
% RST transformation matrix
for ff = 1:fLen
PSI(:,:,ff) = d*exp(1i*(2*pi*f(ff)/c)*MZ.*MM./sqrt(1+MM.^2)) .* exp(-pi*(QZ-QQ).^2/sigma^2);
end    

end

