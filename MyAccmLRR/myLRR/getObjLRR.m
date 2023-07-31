function [objVal] = getObjLRR(X, S, Y, O, lambda, mu, regType)
%
%
objVal = (mu/2)*norm(O*X + Y - O,'fro')^2;

for j = 1 : size(Y,2)
    ccL21(j) = norm(Y(:,j),2);
end
%    
if regType == 1
    nufunobj =  sum(log(S+eps));
    objVal = objVal + lambda*nufunobj + sum(ccL21);  % log(sigma+eps)
else
    nufunobj = sum(sum(log(diag(S).^2+1)));
    objVal = objVal + lambda*nufunobj + sum(ccL21); % log(sigma^2+1)
end

