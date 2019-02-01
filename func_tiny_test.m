function func_tiny_test(roi,tiny,print_statements,plot_mats)
% Regional involvement in cognitive Domain STATS for individual regions


if ~exist('print_statements');print_statements=0;end
if ~exist('plot_mats');plot_mats=0;end

disp(sprintf('ROI:%s',roi))
tmat = [];
for t1 = 1:5
for t2 = 1:5
v1 = squeeze(tiny.mat(ismember(tiny.rlbls,roi),t1,:));
v2 = squeeze(tiny.mat(ismember(tiny.rlbls,roi),t2,:));

%t_thresh = 2.069; % p < .05
t_thresh = 3.104; % p < .005 == bonferroni correction for 10 comparisons
%t_thresh = 3.767; % p < .001

[H,P,CI,STATS] = ttest(v1,v2);
tmat(t1,t2) = STATS.tstat;
    if print_statements
    if ~isnan(H)
    if H & STATS.tstat > 0
    disp(sprintf('%s > %s',tiny.tlbls{t1},tiny.tlbls{t2}))
    t_statement(STATS,P);
    end
    end
    end
    
end
end

    if plot_mats
    add_numbers_to_mat(tmat,tiny.tlbls);
    ylabel('Task 1');
    xlabel('Task 2');
    title({roi 'Task 1 > Task 2'},'fontsize',20);
    set(gca,'CLim',[t_thresh t_thresh+.0001])
    %set(gca,'fontsize',12)
    end