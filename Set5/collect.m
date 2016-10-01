function [Xl, Xh, V_pca] = collect(zooming, im, scale_num, downsample_factor, patch_size, overlap) 
K = 4;
for i = 1 : K
   X{i} = imrotate(im, (i- 1)*90, 'bicubic');   
end
for j = 1 : K
    i = i + 1;
    X{i} = flipud(X{j});
end
Xl = [];
Xh = [];
for i = 1 : length(X)
    [lopatches, hipatches] = collectSamplesMulScales(zooming, X{i}, scale_num, downsample_factor, patch_size, overlap);
    Xl = [Xl, lopatches];
    Xh = [Xh, hipatches];
end
%%
% [Xh, Xl] = patch_pruning(Xh, Xl, 10);
% PCA dimensionality reduction
lores = Xl;
C = double(lores * lores');
[V, D] = eig(C);
D = diag(D); % perform PCA on features matrix 
D = cumsum(D) / sum(D);
k = find(D >= 1e-3, 1); % ignore 0.1% energy
V_pca = V(:, k:end); % choose the largest eigenvectors' projection
Xl = V_pca' * lores;
l2 = sum(Xl.^2).^0.5+eps;
    l2n = repmat(l2,size(Xl,1),1);   
%     l2(l2<0.1) = 1;
Xl = Xl./l2n;
Xh = Xh./repmat(l2,size(Xh,1),1);
Xl = single(Xl);
Xh = single(Xh);
end