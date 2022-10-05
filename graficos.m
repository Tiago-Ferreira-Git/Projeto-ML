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
    plot(to_plot_betas.lasso_vector,to_plot_betas.betas(j,:),'DisplayName',"Beta "+ string(j))
    legend([],'Location','southwest');
    pause(1)
end
hold off
%legend({'Beta 0', 'Beta 1', 'Beta 2', 'Beta 3','Beta 4','Beta 5','Beta 6','Beta 7','Beta 8','Beta 9'},'Location','southwest')
legend show