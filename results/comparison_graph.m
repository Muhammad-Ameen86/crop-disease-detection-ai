% ==========================================
% Crop Disease Detection - Final Comparison
% ==========================================
clc; clear; close all;

models = {'SVM','Random Forest','k-NN', ...
          'AlexNet','ResNet','MobileNet'};

accuracy = [91.93, 83.40, 78.66, 98.16, 83.40, 84.11];

types = categorical({'ML','ML','ML','DL','DL','DL'});
types = reordercats(types, {'ML','DL'});

% --------------------------
% Figure 1: Accuracy Bar Chart
% --------------------------
figure('Name','Model Accuracy Comparison');
bar(accuracy);
set(gca, 'XTickLabel', models, 'XTickLabelRotation', 30);
ylabel('Accuracy (%)');
title('ML vs DL Accuracy Comparison');
ylim([70 100]);
grid on;

for i = 1:length(accuracy)
    text(i, accuracy(i)+0.5, sprintf('%.2f%%', accuracy(i)), ...
        'HorizontalAlignment','center', 'FontWeight','bold');
end

% --------------------------
% Figure 2: ML vs DL Average
% --------------------------
ml_avg = mean(accuracy(1:3));
dl_avg = mean(accuracy(4:6));

figure('Name','Average ML vs DL');
bar([ml_avg dl_avg]);
set(gca, 'XTickLabel', {'Machine Learning','Deep Learning'});
ylabel('Average Accuracy (%)');
title('Average ML vs DL Performance');
ylim([70 100]);
grid on;

text(1, ml_avg+0.5, sprintf('%.2f%%', ml_avg), ...
    'HorizontalAlignment','center', 'FontWeight','bold');
text(2, dl_avg+0.5, sprintf('%.2f%%', dl_avg), ...
    'HorizontalAlignment','center', 'FontWeight','bold');