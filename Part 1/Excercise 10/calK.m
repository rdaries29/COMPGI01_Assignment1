function K = calK(x_train,x_test,sigma)

x = [x_train;x_test];

for ii = 1 : size(x,1)
   for jj = 1 : size(x,1)
       K(ii,jj) = exp(-(norm(x(ii,:)-x(jj,:),2))^2/(2*sigma^2));
   end
end