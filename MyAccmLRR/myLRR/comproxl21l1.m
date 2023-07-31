function E = comproxl21l1(G,lambda,regType)
%
if regType == 1      % l21
    G1 = sqrt(sum(G.^2,1));
    G1(G1==0) = lambda;
    G2 = (G1-lambda)./G1;
    E = G*diag((G1>lambda).*G2);
else                 % l1
    E = zeros(size(G));
    DD = abs(G)-lambda;
    DD2 = DD.*sign(G);
    ID = (DD>0);
    E(ID) = DD2(ID);
end