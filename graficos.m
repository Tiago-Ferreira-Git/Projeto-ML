clear('all')
close('all')
to_plot_mse = load('Projeto/to_plot_lasso_mse.mat');
figure;
plot(to_plot_mse.lasso_vector,to_plot_mse.mean_squared_error_x);
hold on;
xlabel('\lambda');
ylabel('Mean Squared Errors');
grid on;
hold off

to_plot_betas = load('Projeto/to_plot_lasso_betas.mat');
figure;
hold on;
grid on;
for j=1:length(to_plot_betas.betas(:,1))
    xlabel('\lambda');
    ylabel('\beta_{i}');
    %0.05 no primeiro
    if  -1 < to_plot_betas.betas(j,1) && to_plot_betas.betas(j,1) < 1
        plot(to_plot_betas.lasso_vector,to_plot_betas.betas(j,:),'DisplayName',"Beta "+ string(j-1))
        legend([],'Location','northeast');
        pause(1)
    end
end
hold off
legend show



to_plot_loo_score = load('Projeto/to_plot_loo_score.mat');
y = to_plot_loo_score.loo_score;
x = strings(1,16);
for j=1:length(to_plot_loo_score.lasso_vector(:,1))
    if ~strcmp(to_plot_loo_score.lasso_vector(j,:),'linear                          ')
        
        x(j) = sprintf('\\lambda = %0.3f', str2double(to_plot_loo_score.lasso_vector(j,:)));
    else 
        x(j) = to_plot_loo_score.lasso_vector(j,:);
    end
    x(j) = strtrim(x(j));
end
figure
bar(y)
ax = gca;
ax.XTick = [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16]; 
ax.XTickLabels = x;
ax.XTickLabelRotation = 45;
hold on
grid on
ylabel('LOO scores')
hold off

