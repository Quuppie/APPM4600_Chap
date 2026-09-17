%HW3 Ops

A = 0.5*[1 1; 1+10^(-10) 1-10^(-10)];
aInv = inv(A);
graham = transpose(A)*A;
[vectors, values] = eig(graham)

[U,S,V] = svd(A);