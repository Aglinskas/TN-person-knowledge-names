function res = func_plot_dendMat_names(mats,lbls);
% func_plot_dendMat(mats,lbls)
% Function that plots dendrograms, takes in a single matrix, or a cell of matrices (and appropriate lables)

if length(mats)~=length(lbls); error('number of mats doesnt match num of labels');end

res = [];cc = 0;
if iscell(mats);
    numMats = length(mats);
    mats = mats;
else
    numMats = 1;
    mats = {mats};
    lbls = {lbls};
end

for i = 1:numMats;
cc = cc+1;;

sp = subplot(numMats,2,cc);

%mats{i} = atanh(mats{i}); % Fisher Transformation

Y = 1-get_triu(mats{i});
Z = linkage(Y,'ward');
[h x perm] = dendrogram(Z,0,'labels',lbls{i});
ylabel('dissimilarity (a.u)')
[h(1:end).LineWidth] = deal(2);

sp.FontSize = 14;
sp.FontWeight = 'bold';

ord = perm; % get ordering to return
res.ord{i} = ord;
[h(1:end).LineWidth] = deal(3);
xtickangle(45);

cc = cc+1;
sp = subplot(numMats,2,cc);

if length(mats{i}) < 11
    add_numbers_to_mat(mats{i}(ord,ord),lbls{i}(ord));
else
add_numbers_to_mat(mats{i}(ord,ord),lbls{i}(ord),'nonum');
end
colorbar

xtickangle(45)
sp.CLim = [min(get_triu(mats{i})) max(get_triu(mats{i}))];
sp.FontSize = 14;
sp.FontWeight = 'bold';
end % ends matrix loop
end % ends function