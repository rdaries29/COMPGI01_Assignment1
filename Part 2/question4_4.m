%-------------------------------------------------------------------%
% Module: GI01 - Supervised Learning
% Assignment : Coursework 1
% Author : Russel Daries, Nitish Mutha
% Student ID: 16079408, 15113106
% Question: 4 (1 nearest neighbour)
% Section: Part 2
% Description: 1 nearest neighbour sample complexity
% ------------------------------------------------------------------%

clear all;
close all;

% N features
N = 100;
% M samples
M = 10000;
testIterations = 100;

avg_missclassified = zeros(N,testIterations);
m_error = zeros(N,1);
%1-NN
for n = 1:N
    for m = 1:M
        X = 2*randi([0 1], m, n) - 1;
        y = X(:,1);
        
        %test
        missClassified = zeros(1,testIterations);
        for i = 1:testIterations
            X_test = 2*randi([0 1], 100, n) - 1;
            y_test = X_test(:,1);
            closest_Id = 0;
            y_predict = zeros(size(X_test,1),1);
            for j = 1:size(X_test,1)
                dist = sqrt(sum((X - X_test(j,:)).^2,2));
                [val, closest_Id] = min(dist);
                y_predict(j,:) = X(closest_Id,1);
            end
            
            missClassified(1,i) = sum(y_predict~=y_test);
        end
        percentError = mean(missClassified*100/(100));        
        if percentError <= 10
            m_error(n) = m;
            break;
        end
    end
end

figure;
plot(m_error);
set(gcf, 'Color', 'w');
xlabel('n features','FontSize',15);
ylabel('m samples','FontSize',15);
leg=legend('m sample < 10% generalisation error','Location','Best');
set(leg,'FontSize',15);
set(gca,'YMinorTick','on');
grid minor
axis tight;
print('ex4_1nn-final','-depsc');
