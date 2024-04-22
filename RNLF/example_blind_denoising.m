%EXAMPLE_BLIND_DENOISING ( script )
%   Simulate of a blind image denoising example in which the
%   nature of the noise is assumed to be unknown. The noise level
%   function is automatically estimated.
%
%   References
%   ----------
%   Sutour, C., Deledalle, C.-A. & Aujol, J.-F., "Adaptive
%   Regularization of the NL-Means: Application to Image and Video
%   Denoising," Image Processing, IEEE Transactions on , vol.23,
%   no.8, pp.3506,3521, Aug. 2014
%
%   Sutour, C., Deledalle, C.-A. & Aujol, J.-F. "Estimation of the
%   noise level function based on a non-parametric detection of
%   homogeneous image regions." SIAM Journal on Imaging Sciences
%   (in press)
%
%   License
%   -------
%   This work is protected by the CeCILL-C Licence, see
%   - Licence_CeCILL_V2.1-en.txt
%   - Licence_CeCILL_V2.1-fr.txt
%
%   See also RNL, RNLF, NOISE_ESTIMATION.

%   Copyright 2015 Camille Sutour

clear all
close all

addpathrec('.')

% Load image
filename = 'data/lena.png';
img = double(imread(filename));

% Generate noisy image
[img_nse, noisegen] = noisegen(img, 'hybrid', 20);

% Perform blind denoising
param.wait = waitbar(0, 'RNLF denoising...');
[img_rnlf, noise, noise_info] = rnlf(img_nse, param);
close(param.wait);

% Show results
figure('Position', get(0, 'ScreenSize'));
subplot(2, 2, 1);
plotimage(img_nse, img, 'Noisy image');
subplot(2, 2, 2);
plotimage(img_rnlf, img, 'RNLF');
subplot(2, 2, 3);
plothomogeneous(img_nse, noise_info.W, noise_info.hom, img)
subplot(2, 2, 4);
scatterplot(noise_info.stats.m, noise_info.stats.s, ...
            linspace(min(img(:)), max(img(:)), 100), ...
            noise.nlf, noisegen.nlf);
axis square;
legend('Stats', 'Estimated NLF', 'True NLF');
