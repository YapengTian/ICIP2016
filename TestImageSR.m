%% 

%%
close all;clear all; clc;
addpath('Codes');
% set parameters
zooming = 2;                     % scaling factor, depending on the trained dictionary
overlap = 4;
patch_size = 5;
scale_num = 20; 
downsample_factor = 0.98;
%%
im_path = 'Set5/';
im_dir = dir( fullfile(im_path, '*bmp') );
im_num = length( im_dir );
for img = 1:im_num,
X = imread( fullfile(im_path, im_dir(img).name) );
grd = X;
if size(X,3) == 3
    X = rgb2ycbcr(X);
    X = X(:, :, 1);
end
X = modcrop(X, zooming);
grd = modcrop(grd, zooming);
X = double(X);
%%Generate LR image
im_l = imresize(X, 1/zooming, 'bicubic');
[Xl, Xh, V_pca] = collect(zooming, im_l, scale_num, downsample_factor, patch_size, overlap);
fprintf('\n');
N = size(Xl, 2);
K =5000;
idx = randperm(N);
centroids = Xl(:, idx(1:K));
anchors = 2048;
lamda = 0.01;
for i = 1 : K
    c = centroids(:, i);
    dis = abs(c' * Xl);
    [~, id] = sort(dis, 'descend');
    Nl = Xl(:, id(1:anchors));
    Nh = Xh(:, id(1:anchors));
    M{i} = (Nh / (Nl' * Nl +lamda *  eye(size(Nl, 2))))*Nl';
    clear Nl Nh;
end
clear Xl Xh;
x_interp = imresize(im_l, zooming, 'bicubic');
x_bp = backprojection(x_interp, im_l, 20);
 im_h = mcrsr(x_bp, patch_size, centroids, overlap, M, V_pca);
 im_h = backprojection(im_h, im_l, 20);
clear M;
%%
lr = imresize(grd, 1/zooming, 'bicubic');
if size(lr, 3) == 3
    lr = rgb2ycbcr(lr);
    xy = lr(:, :, 1);
    xcb = lr(:, :, 2);
    xcr = lr(:, :, 3);
    bic(:, :, 1) = imresize(xy, zooming, 'bicubic');
    bic(:, :, 2) = imresize(xcb, zooming, 'bicubic');
    bic(:, :, 3) = imresize(xcr, zooming, 'bicubic');
    im_bic = ycbcr2rgb(bic);
    bic(:, :, 1) = uint8(im_h);
    our = ycbcr2rgb(bic);
    lr = ycbcr2rgb(lr);
else
    im_bic = imresize(lr, zooming, 'bicubic');
    our = uint8(im_h);
end
clear bic;
grd = shave(grd, [zooming, zooming]);
our = shave(our, [zooming, zooming]);
im_bic = shave(im_bic, [zooming, zooming]);
%% 
savefile( lr, grd, our,im_bic, im_dir(img).name);
%% 
X = shave(X, [zooming, zooming]);
im_h = shave(im_h, [zooming, zooming]);
x_interp = shave(x_interp, [zooming, zooming]);
bb_psnr = compute_rmse(x_interp, X);
pp_psnr = compute_rmse(X, im_h);
scores(img, 1) = pp_psnr;
scores(img, 2) = ssim(X, im_h);
end
save Set5Result/scores scores;


