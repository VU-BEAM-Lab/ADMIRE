% This code was obtained from the paper "A Tutorial on Independent
% Component Analysis" by Jonathon Shlens. This paper is publicly available 
% on arXiv and can be accessed at https://arxiv.org/pdf/1404.2986.pdf. 
% Paper Citation: Shlens, Jonathon. "A tutorial on independent component 
% analysis." arXiv preprint arXiv:1404.2986 (2014).

function [W, S] = ica(X)
% ICA Perform independent component analysis.
%
% [W, S] = ica(X);
%
% where X = AS and WA = eye(d)
% and [d,n] = size(X) (d dims, n samples).
%
% Implements FOBI algorithm.
%
% JF Cardoso (1989) "Source separation using
% higher order moments", Proc Intl Conf Acoust
% Speech Signal Process
[d, n] = size(X);
% Subtract off the mean of each dimension.
X = X - repmat(mean(X,2),1,n);
% Calculate the whitening filter.
[E, D] = eig(cov(X'));
% Whiten the data
X_w = sqrtm(pinv(D))*E'*X;
% Calculate the rotation that aligns with the
% directions which maximize fourth-order
% correlations. See reference above.
[V,s,u] = svd((repmat(sum(X_w.*X_w,1),d,1).*X_w)*X_w');
% Compute the inverse of A.
W = V * sqrtm(pinv(D)) * E';
% Recover the original sources.
S = W * X;
end