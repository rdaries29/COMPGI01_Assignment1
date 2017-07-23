function lsr = calculateLSR(x,w,n,train_len,test_len)
% Function to calculate y_true, weight vector (w) and resultant MSE for
% train and test vectors.

y = x*w + n;

x_train = x(1:train_len);
x_test = x((length(x)-test_len)+1 : end);

y_train = y(1:train_len);
y_test = y(length(x)-test_len+1 : end);

w_estimator = (x_train'*x_train)\(x_train'*y_train);

mse_train = meanSquareError(w_estimator,x_train,y_train,train_len);
mse_test = meanSquareError(w_estimator,x_test,y_test,test_len);

lsr = [w_estimator, mse_train, mse_test];

end