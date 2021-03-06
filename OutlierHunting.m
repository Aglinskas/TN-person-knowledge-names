clear
load('/Users/aidasaglinskas/Desktop/word_data.mat')
%%
clc
warning('off','stats:linkage:NotEuclideanMatrix')
tpermmat = [];
rpermmat = [];
for i = 1:1000
mat = mean(netRSA.tcmat(:,:,randi(24,1,12)),3);
Z = linkage(squareform(1-mat),'ward');
[Za distmat] = get_Z_atlas(Z);
tpermmat(:,:,i) = distmat;

mat = mean(netRSA.rcmat(:,:,randi(24,1,12)),3);
Z = linkage(squareform(1-mat),'ward');
[Za distmat] = get_Z_atlas(Z);
rpermmat(:,:,i) = distmat;
end
disp('done')
%%
figure(1);
Z = linkage(squareform(mean(rpermmat,3)),'ward');
dendrogram(Z,'labels',rlbls(7:end))

figure(2)
Z = linkage(squareform(mean(tpermmat,3)),'ward');
dendrogram(Z,'labels',tlbls10)
%% ROI outlier

dt = [];
for s = 1:24;

svec = 1:24;
use_vec = svec~=s;


m = mean(netRSA.tcmat(:,:,use_vec),3);
this = netRSA.tcmat(:,:,s);

v1 = squareform(1-this)';
vm = squareform(1-m)';

dt(s) = corr(v1,vm);

end
bar(dt)
%%
s = 0
%%
s = s+1;
svec = [1:size(netRSA.rcmat,3)]~=s;


func_plot_dendMat({mean(netRSA.rcmat(:,:,svec),3) mean(netRSA.tcmat(:,:,svec),3)},{rlbls(7:end) tlbls10})


%% Motion Params 

svec = [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 22 23 24  26 27 28 29 30 31]

% dropped 21 25
s = 1;
sess = 1;
numSubs = length(svec);
temp = '/Users/aidasaglinskas/Google Drive/Aidas/Data_words/S%d/Functional/Sess%d/rp_data.txt';


f= figure(4);

c =0;
disp('running')
for i = 1:numSubs
for sess = 1:5
    c = c+1;
s = svec(i);
    
fn = sprintf(temp,s,sess);
a = load(fn);

sp = subplot(numSubs,5,c);
plot(a(:,1:3));

sp.XTick = [];
xlim([0 210]);
w = [-3 3];
    ylim(w);
    sp.YTick = w;
box off
grid on
if sess==1;
    sp.YLabel.String = num2str(s);
    sp.YLabel.FontSize = 16;
    sp.YLabel.FontWeight = 'bold';
    sp.YLabel.Rotation = 0;
end

end
end
disp('done')
%%
addpath('/Users/aidasaglinskas/Documents/MATLAB/export_fig_fldr/')
ofn = '/Users/aidasaglinskas/Google Drive/Aidas/Data_words/submove_XYZ.pdf'
export_fig(ofn)
%% Reaction times
T = table;
mt_fn_temp = '/Users/aidasaglinskas/Google Drive/Aidas/Data_words/S%d/wS%d_Results.mat';
for subID = 8:31;
mt_fn = sprintf(mt_fn_temp,subID,subID);
myTrials = [];
load(mt_fn);

    e_c = cellfun(@isempty,{myTrials.response})
    [myTrials(e_c).RT] = deal(NaN);
    [myTrials(e_c).response] = deal(NaN);


T.meanRT(subID) = nanmean([myTrials.RT]);
T.RTstd(subID) = nanstd([myTrials.RT]);
T.nSkipped(subID) = sum(e_c);
tab = tabulate({myTrials(~e_c).response});
T.perc1(subID) = tab(1,3);
T.perc2(subID) = tab(2,3);
T.perc3(subID) = tab(3,3);
T.perc4(subID) = tab(4,3);


col = [];
cc_blocks = [11 12 13 14 15 16];
for c = 1:length(cc_blocks)
temp = ismember([myTrials.blockNum],cc_blocks(c));
fmri_blocks = unique([myTrials(temp).fmriBlock]);
    for f = 1:length(fmri_blocks)
    f_inds = find([myTrials.fmriBlock]==fmri_blocks(f));
        for i = 2:8
          hit = strcmp(myTrials(f_inds(i)).word,myTrials(f_inds(i)-1).word) & strcmp(myTrials(f_inds(i)).response,'1');
          corR = ~strcmp(myTrials(f_inds(i)).word,myTrials(f_inds(i)-1).word) & strcmp(myTrials(f_inds(i)).response,'2');
          col(end+1) = hit || corR;
        end

    end
end


T.CC_acc(subID) = mean(col);








end




    
