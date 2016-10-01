%% Anchored Neighborhood Regression Based Single Image Super-Resolution from Self-examples
%% ICIP, 2016
%% By Yapeng Tian, Email: typ14@mails.tsinghua.edu.cn
clc;
clear all;
close all;
addpath('Codes');
%% parameters
overlap = 4;                     % the more overlap the better (patch size 5x5)
zooming = 2;                     % scaling factor
patch_size = 5;
%% Load test image

X = imread('Set5/butterfly_GT.bmp');
if size(X,3) == 3
    X = rgb2ycbcr(X);
    X = X(:, :, 1);
end
X = modcrop(X, zooming);
X = double(X);
%% Generate input LR image 
im_l = imresize(X, 1/zooming, 'bicubic');
%% Learn regression models

%%Extract Training patch pairs
scale_num = 20; 
downsample_factor = 0.98;
[Xl, Xh, V_pca] = collect(zooming, im_l, scale_num, downsample_factor, patch_size, overlap);
%%Ramdom select Anchor Points
N = size(Xl, 2);
K =5000;       % number of anchor points
idx = randperm(N);
centroids = Xl(:, idx(1:K));
%%anchored neighborhood regression
lamda = 0.01;
anchors = 2048;
for i = 1 : K
    c = centroids(:, i);
    dis = abs(c' * Xl);
    [~, id] = sort(dis, 'descend');
    Nl = Xl(:, id(1:anchors));
    Nh = Xh(:, id(1:anchors));
    M{i} = (Nh / (Nl' * Nl +lamda *  eye(size(Nl, 2))))*Nl';
end
%% Test phase;

%%estimate coarse vesion of HR image
x_interp = imresize(im_l, zooming, 'bicubic');
x_bp = backprojection(x_interp, im_l, 20);
%%super-resolution
im_h = mcrsr(x_bp, patch_size, centroids, overlap, M, V_pca);
im_h = backprojection(im_h, im_l, 20);
imwrite(uint8(im_h), 'Result/our.bmp');
%% evaluation 

X = shave(X, [zooming, zooming]);
im_h = shave(im_h, [zooming, zooming]);
x_interp = shave(x_interp, [zooming, zooming]);
pp_psnr = compute_rmse(X, im_h);
fprintf('Our method PSNR = %0.2f\n', pp_psnr);
SSIM = ssim(X, im_h)

