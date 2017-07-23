function mse = meanSquareError(w, x, y, len)
    %Compute MSE for input vectors
    mse = (1/len) * sum(power((x*w - y),2));
    %mse = (1/len) * ((w'*x'*x*w) - (2*y'*x*w) + (y'*y));
end