function [hIm] = mcrsr(mIm,patch_size, centroids_l, overlap, C, U)
%
hIm = zeros(size(mIm));
cntMat = zeros(size(mIm));
[h, w] = size(mIm);
% extract low-resolution image features
lImfea = extr_lIm_fea(mIm);

% patch indexes for sparse recovery (avoid boundary)
gridx = 3:patch_size - overlap : w-patch_size-2;
gridx = [gridx, w-patch_size-2];
gridy = 3:patch_size - overlap : h-patch_size-2;
gridy = [gridy, h-patch_size-2];

cnt = 0;
% loop to recover each low-resolution patch
for ii = 1:length(gridx),
    for jj = 1:length(gridy),
        
        cnt = cnt+1;
        xx = gridx(ii);
        yy = gridy(jj);
        
            mPatchFea = lImfea(yy:yy+patch_size-1, xx:xx+patch_size-1, :);   
            mPatchFea = mPatchFea(:);
            y = U' * mPatchFea;
            %Find The Idx
            dis = abs(y' * centroids_l);
            [~, idx] = max(dis);
            hPatch = C{idx} * y;


            hPatch = reshape(hPatch, [patch_size, patch_size]);
        hIm(yy:yy+patch_size-1, xx:xx+patch_size-1) = hIm(yy:yy+patch_size-1, xx:xx+patch_size-1) + hPatch;
        cntMat(yy:yy+patch_size-1, xx:xx+patch_size-1) = cntMat(yy:yy+patch_size-1, xx:xx+patch_size-1) + 1;
    end
end
% fill in the empty with bicubic interpolation
idx = (cntMat < 1);
cntMat(idx) = 1;
hIm = hIm./cntMat;
hIm = hIm + mIm;
end

