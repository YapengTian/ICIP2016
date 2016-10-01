function [lores, hires] = collectSamplesMulScales(zooming, im, numscales, scalefactor, patch_size, overlap)
lores = [];
hires = [];
im = single(im)/255;
for scale = 1:numscales
    sfactor = scalefactor^(scale-1);
    chires = imresize(im, sfactor, 'bicubic');
    chires = modcrop(chires, zooming);
    clores = imresize(chires, 1/zooming, 'bicubic');
    midres = imresize(clores, zooming, 'bicubic');
    midres = backprojection(midres, clores, 20);
    lImfea = extr_lIm_fea(midres);
    chires = chires - midres;
    % patch indexes for sparse recovery (avoid boundary)
    [h, w] = size(midres);
    gridx = 3:patch_size - overlap : w-patch_size-2;
    gridx = [gridx, w-patch_size-2];
    gridy = 3:patch_size - overlap : h-patch_size-2;
    gridy = [gridy, h-patch_size-2];
    for ii = 1:length(gridx),
            for jj = 1:length(gridy),
                xx = gridx(ii);
                yy = gridy(jj);       
                xl= lImfea(yy:yy+patch_size-1, xx:xx+patch_size-1, :); 
                xl = xl(:);
                xh= chires(yy:yy+patch_size-1, xx:xx+patch_size-1);
                xh = xh(:);
                lores = [lores, xl];
                hires = [hires, xh];
           end
    end
    
   
end


end