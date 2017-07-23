function mse = dualcost(K,y,a)

mse = (K*a - y)'*(K*a - y) / size(y,1);