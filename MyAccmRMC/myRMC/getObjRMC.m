function [objVal] = getObjRMC(X, S, Y, O, A, lambda, mu, regType)
%
%
objVal = (mu/2)*norm((X + Y - O).*A,'fro')^2;

% for j = 1 : size(Y,2)
%     ccL21(j) = norm(Y(:,j),2);
% end
%    
if regType == 1
    nufunobj = sum(sum(log(diag(S)+eps)));
    objVal = objVal + lambda*nufunobj + norm(Y.*A,1);  % log(sigma+eps)
else
    nufunobj = sum(sum(log(diag(S).^2+1)));
    objVal = objVal + lambda*nufunobj + norm(Y.*A,1); % log(sigma^2+1)
end

