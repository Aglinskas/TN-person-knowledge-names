function cmat = func_make_cmat(mat)
% takes in a feature matrix, computes a cmat
%mat = aBeta.fmat
center_data = 1;

    if ndims(mat)~=3;disp(size(mat));error('not a 3d matrix');end
    disp(sprintf('ROIs:%d,Tasks:%d,Subs:%d',size(mat,1),size(mat,2),size(mat,3)));

cmat = {};
for s = 1:size(mat,3)
use_mat = mat(:,:,s);
    
    if center_data
    m1 = use_mat' - nanmean(use_mat');
    m2 = use_mat  - nanmean(use_mat);
    else 
    m1 = use_mat';
    m2 = use_mat;
    end

cmat{1}(:,:,s) = corr(m1,'rows','pairwise');
cmat{2}(:,:,s) = corr(m2,'rows','pairwise');
end
end