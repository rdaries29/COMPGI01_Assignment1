function a = kridgereg(K, y, gamma)

a = (K + (gamma*size(y,1)*eye(size(y,1))) ) \ y;