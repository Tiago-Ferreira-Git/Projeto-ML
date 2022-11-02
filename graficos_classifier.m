
clear('all')
close('all')


f1 = load('Projeto/mat_files/f1_final.mat');
model_names=fieldnames(f1);
fieldname_dims=size(fieldnames(f1),1);
for i=1:fieldname_dims-1
    name=model_names{i};
    y(i) = f1.(name);
end
x = ["SVM(rfb)","SVM(sigm)","SVM(poly)","CNN"]
figure
bar(y)
%plot(y)
ax = gca;
ax.XTick = 1:1:length(x); 
ax.XTickLabels = x;
ax.XTickLabelRotation = 45;

hold on
grid on
ylabel('F1 scores')
hold off

