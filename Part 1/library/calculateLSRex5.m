function [w_estimator, mse_train, mse_valid, mse_test] = calculateLSRex5(x,w,n,train_len,validation_len,test_len, gamma)
% Create y vector
y = x*w + n;

%Segment data sets
% Training data
x_train = x(1:train_len, :);
% Validation set
x_valid = x(train_len+1 : train_len+1+validation_len, :);
% Test Set
x_test = x((length(x)-test_len)+1 : end, :);

% Training set
y_train = y(1:train_len , :);
% Validation set
y_valid = y(train_len+1 : train_len+1+validation_len, :);
% Test set
y_test = y(length(x)-test_len+1 : end, :);

w_estimator = (x_train'*x_train + gamma*train_len*eye(size(w,1)))\(x_train'*y_train);

% Calculation for MSE of each data set
mse_train = meanSquareError(w_estimator,x_train,y_train,train_len);
mse_valid = meanSquareError(w_estimator,x_valid,y_valid,validation_len);
mse_test = meanSquareError(w_estimator,x_test,y_test,test_len);


end