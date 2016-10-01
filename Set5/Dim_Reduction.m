function [X, U1] = Dim_Reduction(x)
%%PCA Dimension Reduction
%   
avg = mean(x, 2);
x = x - repmat(avg, 1, size(x, 2));
sigma =  x * x' / (size(x, 2) - 1);
[U, S, ~] = svd(sigma);
eigval = diag(S);
n = length(eigval);
eig_total = sum(eigval);
eigval = eigval/eig_total;
for i = 1 : n
    s = sum(eigval(1 : i));
    if s >= 0.99
        k = i;
        break;
    end 
end   
U1 = U(:, 1 : k);
X = U1' * x;
end

