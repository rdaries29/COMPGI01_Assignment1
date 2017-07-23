%-------------------------------------------------------------------%
% Module: GI01 - Supervised Learning
% Assignment : Coursework 1
% Author : Russel Daries, Nitish Mutha
% Student ID: 16079408, 15113106
% Question: 4 (Winnow)
% Section: Part 2
% Description: Winnow sample complexity
% ------------------------------------------------------------------%

clear all
close all

N = 100;
M = 1000;
testIterations = 100;
testSize = 500;

m_error = zeros(N,1);
%winnow
for n = 1:N
    for m = 1:M
        X = randi([0 1], m, n);
        y = X(:,1);
        
        %train
        W = ones(1,n);
        for j=1:size(X,1)
            if X(j,:)*W' < n
                y_pred = 0;
            else
                y_pred = 1;
            end
            if sum(y_pred~=y(j,:)) > 0
                W = W .* power(2,(y(j,:)-y_pred)*X(j,:));
            end
        end
        
        %test
        missClassified = zeros(1,testIterations);
        for i = 1:testIterations
            X_test = randi([0 1], testSize, n);
            y_test = X_test(:,1);
            y_predict = X_test*W';
            missClassified(1,i) = sum(y_predict>=n ~= y_test);
        end
        percentError = mean(missClassified)*100/testSize;        
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
leg=legend('m sample < 10% genralisation error','Location','Best');
set(leg,'FontSize',15);
set(gca,'YMinorTick','on');
grid minor
axis tight;
print('ex4_winnow','-depsc');