function func_grouped_errorbar(model_series,model_error,sig)



plot_sig = 0;
if exist('sig');plot_sig=1;end

ax = axes;
h = bar(model_series,'BarWidth',1);
hold on;
% Finding the number of groups and the number of bars in each group
ngroups = size(model_series, 1);
nbars = size(model_series, 2);
% Calculating the width for each bar group
groupwidth = min(0.8, nbars/(nbars + 1.5));
% Set the position of each error bar in the centre of the main bar
% Based on barweb.m by Bolu Ajiboye from MATLAB File Exchange

colX = []; % collect x
for i = 1:nbars % loop task bars
    % Calculate center of each bar
    x = (1:ngroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*nbars);
    colX(i,:) = x;
    errorbar(x, model_series(:,i), model_error(:,i), 'k', 'linestyle', 'none');

text_y = 0;
    if plot_sig
        for sig_th = 1:size(sig,3)
                for ii = 1:length(x) % goes to 9
                    if sig(ii,i,sig_th) % put an X if it's significant
                    t = text(x(ii)-.06,text_y,'*','fontsize',26);
                    %disp(t.Extent)
                   %text(1,t.Extent(2) + t.Extent(4),'i')
                    end
                end
                text_y = [t.Extent(2) + t.Extent(4)];
        end
    end
end


plot_mean_line = 1;
mean_roi_resp = mean(model_series,2);
for i = 1:size(colX,2) 
which_x = colX(:,i);

which_x(1) = which_x(1) - (groupwidth / 5 / 2);
which_x(end) = which_x(end) + (groupwidth / 5 / 2);

which_y = repmat(mean_roi_resp(i),1,length(which_x));
plot(which_x,which_y,'k--','LineWidth',2)
end



