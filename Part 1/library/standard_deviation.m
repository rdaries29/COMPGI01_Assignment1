function sd = standard_deviation(avg_error,test_errors)
%Compute the standard deviation of inputs

[dim1, dim2] = size(test_errors);

sum_of_sq = 0;
for i=1:dim1
    sum_of_sq = sum_of_sq + power((test_errors(i) - avg_error),2);
end
%Output the resultant standard deviation
sd = sqrt(sum_of_sq / dim1);