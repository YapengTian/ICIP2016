function [] = savefile(imLR, ori_HR, im_rgb,imbicubic, filename)
    
    filename = filename(1:end-4);
    imLR = imresize( ori_HR, 1.0/3, 'Bicubic');
    imwrite(imLR,  ['Result\', filename, '_LR.bmp']);
    imwrite(ori_HR,  ['Result\', filename, '_HR.bmp']);
    imwrite(im_rgb,  ['Result\', filename, '_our.bmp']);
    imwrite(imbicubic,  ['Result\', filename, '_bicubic.bmp']);
end