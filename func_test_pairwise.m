function func_test_pairwise(dt,lbls,print_statements,plot_mats)

if ~exist('print_statements');print_statements=0;end
if ~exist('plot_mats');plot_mats=0;end

tmat = [];
for t1 = 1:size(dt,2)
for t2 = 1:size(dt,2)

v1 = dt(:,t1);
v2 = dt(:,t2);

t_thresh = 2.069; % p < .05
%t_thresh = 3.104; % p < .005 == bonferroni correction for 10 comparisons
%t_thresh = 3.767; % p < .001

[H,P,CI,STATS] = ttest(v1,v2);
tmat(t1,t2) = STATS.tstat;
    if print_statements
    if ~isnan(H)
    if H & STATS.tstat > 0
    disp(sprintf('%s > %s',lbls{t1},lbls{t2}))
    t_statement(STATS,P);
    end
    end
    end
    
end
end

    if plot_mats
    add_numbers_to_mat(tmat,lbls);
    ylabel('1');
    xlabel('2');
    title({'1 > 2'},'fontsize',20);
    set(gca,'CLim',[t_thresh t_thresh+.0001])
    %set(gca,'fontsize',12)
    end