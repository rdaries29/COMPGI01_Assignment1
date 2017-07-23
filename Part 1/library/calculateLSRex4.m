function [w_estimator, mse_train, mse_test] = calculateLSRex4(x,w,n,train_len,test_len, gamma)

y = x*w + n;

% Data segmentation and slicing
x_train = x(1:train_len, :);
x_test = x((length(x)-test_len)+1 : end, :);

y_train = y(1:train_len , :);
y_test = y(length(x)-test_len+1 : end, :);

% Compute weight vectors
w_estimator = (x_train'*x_train + gamma*train_len*eye(size(w,1)))\(x_train'*y_train);

% Compute MSE for train and test sets
mse_train = meanSquareError(w_estimator,x_train,y_train,train_len);
mse_test = meanSquareError(w_estimator,x_test,y_test,test_len);


end