%-------------------------------------------------------------------%
% Module: GI01 - Supervised Learning
% Assignment : Coursework 1
% Author : Russel Daries, Nitish Mutha
% Student ID: 16079408, 15113106
% Question: 4 (Least squares)
% Section: Part 2
% Description: Least Squares sample complexity
% ------------------------------------------------------------------%

clear all
close all

% N feaures
N = 100;
% M samples
M = 500;
testIterations = 100;
testSize = 500;

avg_missclassified = zeros(testIterations,N);
m_error = zeros(N,1);
%least square
for n = 1:N
    for m = 1:M
        
        %train
        X = 2*randi([0 1], m, n) - 1;
        y = X(:,1);
        W = pinv(X)*y;
        
        %test
        missClassified = zeros(1,testIterations);
        for i = 1:testIterations
            X_test = 2*randi([0 1], testSize, n) - 1;
            y_test = X_test(:,1);
            y_predict = sign(X_test*W);
            missClassified(1,i) = sum(y_predict~=y_test);
        end
        percentError = mean(missClassified*100/testSize);        
        if percentError <= 10
            m_error(n) = m;
            avg_missclassified(:,n) = missClassified';
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
print('ex4_least_sq','-depsc');