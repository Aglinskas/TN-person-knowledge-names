%% Load Data
loadMR
cd '/Users/aidasaglinskas/Google Drive/Aidas/Data_words/Scripts/fMRI-person-knowledge-names';
%% Format Labels
rlbls = aBeta.r_lbls;
    rlbls = strrep(rlbls,'.mat','');
    rlbls = strrep(rlbls,'-left','-L');
    rlbls = strrep(rlbls,'-right','-R');
    rlbls = strrep(rlbls,'Angular','AG');
    rlbls = strrep(rlbls,'Precuneus','PREC');
    rlbls = strrep(rlbls,'Amygdala','AMY');
tlbls10 = aBeta.t_lbls(1:10);
%% Ordering of ROIs and Tasks for visualisation
%r_ord = [13 14 9 10 19 20 3 4 17 18 21 7 8 11 12 15 16 5 6 1 2];
r_ord = [13 14 9 10 19 20 11 12 15 16 1 2 5 6 3 4 7 8 17 18 21];
t_ord = [3 4 2 9 1 5 7 8 6 10];
%%
%r_ord = 1:21
%t_ord = 1:10
aBeta.wmat = aBeta.wmat(r_ord,t_ord,:);
aBeta.wmat_raw = aBeta.wmat_raw(r_ord,[t_ord 11 12],:);
aBeta.fmat = aBeta.fmat(r_ord,t_ord,:);
aBeta.fmat_raw = aBeta.fmat_raw(r_ord,[t_ord 11 12],:);
rlbls = rlbls(r_ord);
tlbls10 = tlbls10(t_ord);
aBeta.r_lbls = rlbls;
%% Names > Monuments
figure(2);clf
m = aBeta.wmat_raw(:,11,:) - aBeta.wmat_raw(:,12,:);
m = squeeze(m);

func_plot_tbar_plot(m',rlbls,1); % Bonferroni Corrected
title('Person Names 1-back > Monument Names 1-back','fontsize',20);
set(gca,'FontSize',14);

[H,P,CI,STATS] = ttest(m(ismember(rlbls,'PREC'),:),0,'tail','right');
t_statement(STATS,P);
%% Person Knowledge > Name 1-back
figure(2);clf
m = mean(aBeta.wmat,2);
m = squeeze(m);

func_plot_tbar_plot(m',rlbls,1) % Bonf corrected
title('Mean all tasks > name -back','fontsize',20)
set(gca,'FontSize',14)
%% Face + Name bar plot 
rlbls = aBeta.r_lbls;
mat1 = squeeze(mean(aBeta.fmat,2));
mat2 = squeeze(mean(aBeta.wmat,2));

r_start = 7
mat1 = mat1(r_start:end,:);
mat2 = mat2(r_start:end,:);
rlbls = rlbls(r_start:end);
m1 = mean(mat1,2);
m2 = mean(mat2,2);
m3 = zeros(size(m2));

se1 = std(mat1,[],2) ./ sqrt(size(mat1,2));
se2 = std(mat2,[],2) ./ sqrt(size(mat2,2));
se3 = zeros(size(se2));
FigID = figure(2); clf

addpath('/Users/aidasaglinskas/Documents/MATLAB/errorbar_groups/')
try
errorbar_groups([m1 m2 m3 ]',[se1 se2 se3 ]','FigID',FigID,'bar_width',1.2,'errorbar_width',4,'optional_errorbar_arguments',{'LineStyle','none','Marker','none','LineWidth',1.2})
catch
end

FigID.Color = [1 1 1]
FigID.CurrentAxes.XTick = 1.5 : 3 : 44;
FigID.CurrentAxes.XTickLabel = rlbls;
FigID.CurrentAxes.XTickLabelRotation = 45;
FigID.CurrentAxes.FontSize = 12
FigID.CurrentAxes.FontWeight = 'bold'

title('Regional involvement in person knowledge','fontsize',20)


% 
xvec = .5:3:43; 
[H,P,CI,STATS] = ttest(mat1',0,'alpha',.05/size(mat1,1));
xvec = xvec(logical(H));
text(xvec-.2,zeros(size(xvec)),'*','fontsize',30);

xvec = 1.5:3:46;
[H,P,CI,STATS] = ttest(mat2',0,'alpha',.05/size(mat1,1));
xvec = xvec(logical(H));
text(xvec,zeros(size(xvec)),'*','fontsize',30);
xlim([-2 46])
ylim([-.3 5.5])
l = legend({'Faces' 'Names' 'p < .05/15 (Bonf. corr.)'},'location','best')
l.FontSize = 14
ylabel('Response magnitude (β)')
%% Make tiny matrix: Averaged across task domains

task_groups = {{'Friendliness' 'Trustworthiness'} {'Attractiveness' 'Distinctiveness'} {'First memory' 'Familiarity'} {'How many facts' 'Occupation'} {'Common name' 'Full name'}};
task_groups_leg = {'Social' 'Physical' 'Episodic' 'Biographical' 'Nominal'};

%roi_groups = {{'OFA-L' 'OFA-R'} {'FFA-L' 'FFA-R'} {'pSTS-L' 'pSTS-R'} {'ATL-L' 'ATL-R'} {'PREC'} {'dmPFC'} {'vmPFC'} {'AG-L' 'AG-R'} {'IFG-L' 'IFG-R'} {'OFC-L' 'OFC-R'} {'AMY-L' 'AMY-R'} {'ATFP-L' 'ATFP-R'}};
%roi_groups_leg = {'OFA' 'FFA' 'pSTS' 'ATL' 'PREC' 'dmPFC' 'vmPFC' 'AG' 'IFG' 'OFC' 'AMY' 'ATFP'};

roi_groups = {{'OFA-L' 'OFA-R'} {'FFA-L' 'FFA-R'} {'pSTS-L' 'pSTS-R'} {'IFG-L' 'IFG-R'} {'OFC-L' 'OFC-R'} {'AMY-L' 'AMY-R'} {'ATFP-L' 'ATFP-R'}  {'ATL-L' 'ATL-R'} {'PREC'} {'dmPFC'} {'vmPFC'} {'AG-L' 'AG-R'} };
roi_groups_leg = {'OFA' 'FFA' 'pSTS' 'IFG' 'OFC' 'AMY' 'ATFP' 'ATL' 'PREC' 'dmPFC' 'vmPFC' 'AG'};


tiny_mat = [];
rlbls = aBeta.r_lbls;
for r = 1:length(roi_groups_leg)
for t = 1:length(task_groups_leg)

v = aBeta.wmat(ismember(rlbls,roi_groups{r}),ismember(tlbls10,task_groups{t}),:);
tiny_mat(r,t,:) = mean(mean(v,1),2);

end
end


tiny.mat = tiny_mat;
tiny.rlbls = roi_groups_leg;
tiny.tlbls = task_groups_leg;


% Trim away core regions
tiny.rlbls(1:3) = [];
tiny.mat(1:3,:,:) = [];


tiny
% Plot tiny matrix
%tiny.mat = tiny.mat - mean(tiny.mat,2); % remove regional mean, cleaner tuning profiles


m = mean(tiny.mat,3);
s = std(tiny.mat,[],3);
s = s ./ sqrt(size(tiny.mat,3));
sig = [];
sig(:,:,1) = ttest(tiny.mat,0,'dim',3,'alpha',.05);
sig(:,:,2) = ttest(tiny.mat,0,'dim',3,'alpha',.005);
sig(:,:,3) = ttest(tiny.mat,0,'dim',3,'alpha',.001);


% Normalisation, for each sub -mean ./std
norm = 0;
    if norm
for r = 1:size(tiny_mat,1)
for s = 1:size(tiny_mat,3)
tiny_mat(r,:,s) = tiny_mat(r,:,s) - mean(tiny_mat(r,:,s));
tiny_mat(r,:,s) = tiny_mat(r,:,s) ./ std(tiny_mat(r,:,s));
end
end
tiny.mat = tiny_mat;
tiny.rlbls = roi_groups_leg;
tiny.tlbls = task_groups_leg;
tiny.rlbls(1:3) = [];
tiny.mat(1:3,:,:) = [];
m = mean(tiny.mat,3);
s = std(tiny.mat,[],3);
s = s ./ sqrt(size(tiny.mat,3));
    end



%
f = figure(3);clf
%plot --size 1500,1000
 model_series = m;
 model_error = s;
sig = sig;
func_grouped_errorbar(m,s,sig)
xticklabels(tiny.rlbls);
l_entries = tiny.tlbls;
l_entries{end+1} = '* < .05'
l_entries{end+1} = '** < .005'
l_entries{end+1} = '*** < .001'
l_entries{end+1} = '-- ROI mean'
l = legend(l_entries,'location','best');
box off
set(gca,'fontsize',20);
title({'Regional Involvement in task domains'})
ylabel('Response magnitude (β)');
xtickangle(45)
f.CurrentAxes.FontWeight = 'BOLD'
f.Color = [1 1 1];
%plot --size 1000,600
%func_plot_tbar_plot(squeeze(aBeta.wmat(ismember(rlbls,'ATFP-L'),:,:))',tlbls10,1)

%%
figure(4)
subplot(2,2,1)
func_tiny_test('PREC',tiny,0,1)

subplot(2,2,2)
func_tiny_test('AG',tiny,1,1)

subplot(2,2,3)
func_tiny_test('AMY',tiny,0,1)

subplot(2,2,4)
func_tiny_test('ATFP',tiny,0,1)
%% NetRSA set-up

netRSA.mat = aBeta.wmat;
netRSA.fmat = aBeta.fmat;
netRSA.rlbls = rlbls;
netRSA.tlbls = tlbls10;

% Trim Matrix to drop Core Rois
roi_to_drop = {};
roi_to_drop = {'OFA-L' 'OFA-R' 'FFA-L' 'FFA-R' 'pSTS-L' 'pSTS-R'};
%roi_to_drop = {'OFA-L' 'OFA-R' 'FFA-L' 'FFA-R' 'pSTS-L' 'pSTS-R' 'ATFP-R' 'ATFP-L' 'AMY-L' 'AMY-R' 'ATL-R' 'IFG-R'};
drop_inds = ismember(netRSA.rlbls,roi_to_drop);

netRSA.mat(drop_inds,:,:) = [];
netRSA.fmat(drop_inds,:,:) = [];
netRSA.rlbls(drop_inds) = [];
%%

cmats = func_make_cmat(netRSA.mat);
netRSA.rcmat = cmats{1};
netRSA.tcmat = cmats{2};


c = func_make_cmat(netRSA.fmat);
netRSA.fr_model = mean(c{1},3);
netRSA.ft_model = mean(c{2},3);

netRSA
%%
figure(4)
%plot --size 2000,1500
in_mat = netRSA.mat;
cmats = func_make_cmat(in_mat); % Make similarity matrices
res = func_plot_dendMat_names({mean(netRSA.rcmat,3) mean(netRSA.tcmat,3)},{netRSA.rlbls netRSA.tlbls}); %Plot dendrograms
%% NetRSA regional models

figure(5);clf
%func_made_RSA_model
%func_fit_RSA_model
clust = {};
clust_ttl = {};
l = 0;
%l = l+1;clust{l} = {{}{}}


l = l+1;
clust_ttl{l} = 'Pairs';
clust{l} = {{'ATL-L'    'ATL-R'}  {'AG-L'    'AG-R'}  {'IFG-L'    'IFG-R'} {'OFC-L'    'OFC-R'} {'AMY-L'    'AMY-R'}    {'ATFP-L' 'ATFP-R'}  'PREC'    'dmPFC'    'vmPFC'};

l = l+1;
clust_ttl{l} = 'PostHoc';
clust{l} = {{'ATL-L'    'ATL-R'    'PREC'    'dmPFC'    'vmPFC'    'AG-L'    'AG-R' 'IFG-L'    'IFG-R' } { 'AMY-L'    'AMY-R'    'ATFP-L' 'ATFP-R' 'OFC-L'    'OFC-R'}};

l = l+1;
clust_ttl{l} = 'DMN vs AMY+ATFP vs OFC+IFG';
clust{l} = {{'ATL-L'    'ATL-R'    'PREC'    'dmPFC'    'vmPFC'    'AG-L'    'AG-R'} { 'AMY-L'    'AMY-R'    'ATFP-L' 'ATFP-R'}  {'IFG-L'    'IFG-R' 'OFC-L'    'OFC-R'}};


nmodels = length(clust);


model = [];
show_model = 1;


for i = 1:nmodels
model(:,:,i) = func_made_RSA_model(netRSA.rlbls,clust{i});

    if show_model
    subplot(ceil(length(clust)/2),2,i);
    add_numbers_to_mat(model(:,:,i),netRSA.rlbls,'nonum');
    xtickangle(45)
    title([num2str(i) ': ' clust_ttl{i}],'fontsize',16)
    end
end


model(:,:,end+1) = netRSA.fr_model;
clust_ttl{end+1} = 'Face Model'

% Model Fit Bar
figure(6)
dt = [];
for i = 1:size(model,3);
dt(:,i) = func_fit_RSA_model(netRSA.rcmat,model(:,:,i));
end

func_plot_tbar_plot(dt,clust_ttl)
title('Model Fit','fontsize',16)
ylabel('Model-Data correlation (r)')

rmodel = model;
rdt = dt;
% Model Fit Matrix
figure(7)
func_test_pairwise(dt,clust_ttl,0,1);
title('Model Comparison','fontsize',16);
%%
[H,P,CI,STATS]= ttest(dt(:,1));
t_statement(STATS,P);
%%
[H,P,CI,STATS]= ttest(dt(:,3),dt(:,1));
t_statement(STATS,P);
%clust_ttl{1}
%% netRSA: Cognitive Models


clust = {};
clust_ttl = {};
l = 0;

%l = l+1;
%clust_ttl{l} = '';
%clust{l} = {};


l = l+1;
clust_ttl{l} = 'Pairs';
clust{l} = {{ 'Friendliness'    'Trustworthiness'} {'Attractiveness'  'Distinctiveness' } {'First memory'    'Familiarity'} { 'How many facts' 'Occupation'} {'Common name'    'Full name'}};


l = l+1;
clust_ttl{l} = 'Pairs PostHoc';
clust{l} = {{ 'Friendliness'    'Trustworthiness'} {'Attractiveness'} {'First memory'    'Familiarity'} { 'How many facts' 'Occupation'} {'Common name'    'Full name' 'Distinctiveness'}};

% l = l+1;
% clust_ttl{l} = 'Social-Physical vs Episodic+Biographical vs Nominal';
% clust{l} = {{  'Friendliness'    'Trustworthiness'    'Attractiveness' 'Distinctiveness'} {'First memory'    'Familiarity'    'How many facts' 'Occupation'} { 'Common name'    'Full name'}};


l = l+1;
clust_ttl{l} = 'Episodic+Biographical vs Nominal';
clust{l} = {{ 'Friendliness'    'Trustworthiness'} {'Attractiveness'  'Distinctiveness' } {'First memory'    'Familiarity' 'How many facts'    'Occupation' } {'Common name'    'Full name'}};


l = l+1;
clust_ttl{l} = 'Episodic+Nominal vs Biographical';
clust{l} = { { 'Friendliness'    'Trustworthiness'} {'Attractiveness'  'Distinctiveness' } {'First memory'    'Familiarity'} {'How many facts'    'Occupation' 'Common name'    'Full name'}};


l = l+1;
clust_ttl{l} = 'Social+Physical vs Episodic+Biographical vs Nominal';
clust{l} = {{ 'Friendliness'    'Trustworthiness' 'Attractiveness'  'Distinctiveness' } {'First memory'    'Familiarity' 'How many facts'    'Occupation' } {'Common name'    'Full name'}};



nmodels = length(clust)

figure(8);clf
%plot --size 1000,1500
model = [];
show_model = 1;
for i = 1:nmodels
model(:,:,i) = func_made_RSA_model(netRSA.tlbls,clust{i});

    if show_model
    subplot(ceil(length(clust)/2),2,i);
    add_numbers_to_mat(model(:,:,i),netRSA.tlbls,'nonum');
    xtickangle(45)
    title([num2str(i) ': ' clust_ttl{i}],'fontsize',16)
    end
end


model(:,:,end+1) = netRSA.ft_model;
clust_ttl{end+1} = 'Face Model';
%
figure(9);clf
dt = [];
for i = 1:size(model,3);
dt(:,i) = func_fit_RSA_model(netRSA.tcmat,model(:,:,i));
end

func_plot_tbar_plot(dt,clust_ttl)
title('Model Fit','fontsize',16)
ylabel('Model-Data correlation (r)')
%
figure(10);clf
func_test_pairwise(dt,clust_ttl,1,1);
title('Model Comparison','fontsize',16);
%% 
%[H,P,CI,STATS] = ttest(dt(:,6))
[H,P,CI,STATS] = ttest(dt(:,5),dt(:,6))
t_statement(STATS,P)
%% ROI clustering figure.




%% MD Scale of Tasks
f = figure(1);clf;hold on
i = 1;

c = mean(netRSA.tcmat,3);
md = mdscale(c,2);
md = md(:,[2 1]);
col = [0 0 1;0 0 1;1 .7 0;1 .7 0;1 0.1 .7;1 0.1 .7;.5 0 1;.5 0 1;0 .6 0;0 .6 0];
col = col * .8
for i = 1:10
plot(md(i,1),md(i,2),'o','markersize',20,'MarkerFaceColor',col(i,:),'MarkerEdgeColor',col(i,:))
text(md(i,1)+.02,md(i,2),netRSA.tlbls{i},'FontSize',20,'color',col(i,:),'fontweight','bold');
end

f.CurrentAxes.LineWidth = 2;
f.CurrentAxes.FontSize = 14;
f.CurrentAxes.FontWeight = 'bold';
f.Color = [1 1 1]


xlim([min(md(:,1))-.1  max(md(:,1))+.25])
ylim([min(md(:,2))-.1  max(md(:,2))+.1])
xlabel('Distance in MD dimension 1 (a.u)')
ylabel('Distance in MD dimension 2 (a.u)')
title('Task Similarity','fontsize',30)
box on




plt_list = [1 0.1 .7	 % Episodic
.5 0 1 % Factual
0 0 1 % Social
1 .7 0 % Physical
0 .6 0] % Nominal

f.Position = [ 768        1077         784        1250];
%% Plot Task models
f = figure(11);clf
for i = 1:size(model,3)
sp = subplot(3,2,i)
pl_mat = model(:,:,i);
add_numbers_to_mat(pl_mat,'nonum');
sp.CLim = [[min(1-squareform(1-pl_mat)) max(1-squareform(1-pl_mat))]]
sp.TickDir = 'out'
sp.XTick = [];
sp.YTick = [];
sp.LineWidth = 2
end
f.Color = [1 1 1];
f.CurrentAxes.Color = [1 0 0]
%% Task Model Coparison Bars

d = [dt(:,1) dt(:,3) dt(:,5)];
m = mean(d);
se = std(d) ./ sqrt(size(d,1));
f = figure(12);clf;hold off;

hb = bar(m);hold on;
he = errorbar(m,se,'r.');hold off

hb.LineWidth = 3;
he.LineWidth = 3;
box off

ylabel('Correlation with model (r)')
f.CurrentAxes.LineWidth = 3;
f.CurrentAxes.FontSize = 16;
f.CurrentAxes.FontWeight = 'bold';
f.CurrentAxes.TickDir = 'out';
%% MD Scale OF rois

in_m = mean(netRSA.rcmat,3);
md = mdscale(1-in_m,2);
figure(13);

in_l = netRSA.rlbls;
for i = 1:length(md)
   
    
text(md(i,1),md(i,2),in_l{i})
    
end

xlim([min(md(:,1)) max(md(:,1))])
ylim([min(md(:,2)) max(md(:,2))])










%% Plot ROI models
f = figure(11);clf
for i = 1:size(rmodel,3)
sp = subplot(1,4,i)
pl_mat = rmodel(:,:,i);
add_numbers_to_mat(pl_mat,'nonum');
sp.CLim = [[min(1-squareform(1-pl_mat)) max(1-squareform(1-pl_mat))]]
sp.TickDir = 'out'
sp.XTick = [];
sp.YTick = [];
sp.LineWidth = 2
end
f.Color = [1 1 1];
f.CurrentAxes.Color = [1 0 0]
%% ROI Model Coparison Bars

d = [rdt(:,1) rdt(:,2) rdt(:,3) rdt(:,4)];
m = mean(d);
se = std(d) ./ sqrt(size(d,1));
f = figure(12);clf;hold off;

hb = bar(m);hold on;
he = errorbar(m,se,'r.');hold off

hb.LineWidth = 3;
he.LineWidth = 3;
box off

ylabel('Correlation with model (r)')
f.CurrentAxes.LineWidth = 3;
f.CurrentAxes.FontSize = 16;
f.CurrentAxes.FontWeight = 'bold';
f.CurrentAxes.TickDir = 'out';
f.Color = [1 1 1]
%%

model1 = mean(netRSA.rcmat,3);
model2 = netRSA.fr_model;

dt = [];
dt(:,1) = func_fit_RSA_model(netRSA.rcmat,model1);
dt(:,2) = func_fit_RSA_model(netRSA.rcmat,model2);

figure(14);clf;hold off;
func_plot_tbar_plot(dt,{'wmodel' 'fmodel'})

[H,P,CI,STATS] = ttest(dt(:,1),dt(:,2));
t_statement(STATS,P);

%%
model1 = mean(netRSA.tcmat,3);
model2 = netRSA.ft_model;

dt = [];
dt(:,1) = func_fit_RSA_model(netRSA.tcmat,model1);
dt(:,2) = func_fit_RSA_model(netRSA.tcmat,model2);

figure(14);clf;hold off;
func_plot_tbar_plot(dt,{'wmodel' 'fmodel'})

[H,P,CI,STATS] = ttest(dt(:,1),dt(:,2));
t_statement(STATS,P);
%% Regional stability in core vs extended systems


core = {'OFA-L' 'OFA-R' 'FFA-L' 'FFA-R' 'pSTS-L' 'pSTS-R'};
extended = {'IFG-L' 'IFG-R' 'OFC-L' 'OFC-R' 'ATFP-L' 'ATFP-R' 'AMY-L' 'AMY-R' 'ATL-L' 'ATL-R' 'AG-L' 'AG-R' 'PREC' 'dmPFC' 'vmPFC'};

core_inds = ismember(aBeta.r_lbls,core);
extended_inds = ismember(aBeta.r_lbls,extended);


model = func_make_cmat(aBeta.fmat(core_inds,:,:));
cmats = func_make_cmat(aBeta.wmat(core_inds,:,:));
model_fit_core = func_fit_RSA_model(cmats{2},mean(model{2},3));

model = func_make_cmat(aBeta.fmat(extended_inds,:,:));
cmats = func_make_cmat(aBeta.wmat(extended_inds,:,:));
model_fit_ext = func_fit_RSA_model(cmats{2},mean(model{2},3));

[H,P,CI,STATS] = ttest(model_fit_core,model_fit_ext);
t_statement(STATS,P);


mat = [model_fit_ext model_fit_core];
m = mean(mat);
se = std(mat) ./ sqrt(length(mat));
%% Plot Bar 
f = figure(1);clf
bar(m);hold on;
errorbar(m,se,'r.');hold off
f.Color = [1 1 1];
f.CurrentAxes.FontSize = 16
f.CurrentAxes.FontWeight = 'bold';
f.CurrentAxes.LineWidth = 1;
xticks([])
box off
%%
%res = func_plot_dendMat_names({mean(netRSA.rcmat,3) mean(netRSA.tcmat,3)},{netRSA.rlbls netRSA.tlbls}); %Plot dendrograms

Y = mean(netRSA.tcmat,3);
Y = squareform(1-Y);
Z = linkage(Y,'ward');

%%

f = figure(1);clf
[h x perm] = dendrogram(Z,'labels',netRSA.tlbls);
[h(1:end).LineWidth] = deal(4);
[h.Color] = deal([0 0 0])
xtickangle(45)
ax = gca;
ax.XTickLabel = {'\color[rgb]{0,0,1}Friendliness'
'\color[rgb]{0,0,1}Trustworthiness'
'\color[rgb]{.9,.6,.3}Attractiveness'
'\color[rgb]{.9,.6,.3}Distinctiveness'
'\color[rgb]{0,.6,0}Full name'
'\color[rgb]{0,.6,0}Common name'
'\color[rgb]{1,.1,.7}First memory'
'\color[rgb]{.5,0,1}How many facts'
'\color[rgb]{1,.1,.7}Familiarity'
'\color[rgb]{.5,0,1}Occupation'}

f.CurrentAxes.FontSize = 14;
f.CurrentAxes.FontWeight = 'bold';




















plt_list = [1 0.1 .7	 % Episodic
.5 0 1 % Factual
0 0 1 % Social
1 .7 0 % Physical
0 .6 0] % Nominal


%%

r_inds = extended_inds;
%r_inds = 1:21
cmat = func_make_cmat(aBeta.fmat(r_inds,:,:));

Y = squareform(1-mean(cmat{1},3));
Z = linkage(Y,'ward');

f= figure(1);clf
[h x perm] = dendrogram(Z,'labels',aBeta.r_lbls(r_inds))

[h(1:end).LineWidth] = deal(2);
[h(1:end).Color] = deal([0 0 0]);
f.CurrentAxes.FontSize = 14;
f.CurrentAxes.FontWeight = 'Bold';
f.CurrentAxes.XTickLabelRotation = 45;
%%