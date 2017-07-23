function [w_estimator, mse_test] = calculateLSRTest(x,w,n,train_len,validation_len,test_len, gamma)

y = x*w + n;

x_train = x(1:(train_len+validation_len), :);
x_valid = x(train_len+1 : train_len+1+validation_len, :);
x_test = x((length(x)-test_len)+1 : end, :);

y_train = y(1:(train_len+validation_len) , :);
y_valid = y(train_len+1 : train_len+1+validation_len, :);
y_test = y(length(x)-test_len+1 : end, :);

w_estimator = (x_train'*x_train + gamma*train_len*eye(size(w,1)))\(x_train'*y_train);

%mse_train = meanSquareError(w_estimator,x_train,y_train,train_len);
%mse_valid = meanSquareError(w_estimator,x_valid,y_valid,validation_len);
mse_test = meanSquareError(w_estimator,x_test,y_test,test_len);


end