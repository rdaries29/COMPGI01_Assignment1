function w_estimator = learnModel(x,y)

w_estimator = (x'*x)\(x'*y);

end