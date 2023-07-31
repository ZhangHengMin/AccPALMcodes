function [ U, S, V ] = myGSVT( Z, lambda, regType, rnk )
%% ------------------------------------------------------------------------
% exact solve low rank proximal step
% (1/2)*|X - Z|_F^2 + lambda |X|_theta
%% ------------------------------------------------------------------------
%  regtype = 1: % log(sigma+eps)
%            2: % log(sigma^2+1)
%            3: TNN
%% ------------------------------------------------------------------------
if(exist('rnk', 'var'))
    [U, S, V] = lansvd(Z, rnk, 'L');
else
    [U, S, V] = svd(Z, 'econ');
end
s = diag(S);
switch(regType)
    case 1 % log(sigma+eps)
        s = proximal_log1(s, lambda, eps);
    case 2 % log(sigma^2+1)
        s = proximal_log2(s, length(s), lambda);
        %     case 3 % TNN
        %         s = proximalRegC(s, length(s), lambda, theta, 3);
    otherwise
        assert(false);
end
svs = sum(s > 1e-10);
U = U(:,1:svs);
V = V(:,1:svs);
S = diag(s(1:svs));

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1-log(sigma+eps)
function sigmaX = proximal_log1(s,C,cc)
%
temp = (s-cc).^2 - 4*(C-cc*s);
ind = find (temp>0);
%svp = length(ind);
sigmaX = max(s(ind)-cc+sqrt(temp(ind)),0)/2;
%
end
%
% 2-log(sigma^2+1)
function sigmaX = proximal_log2(s,r,rho)
%
P = [rho*ones(r,1), -1*rho*s, (rho+1)*ones(r,1), -1*rho*s];
rt = zeros(r,1);
for t = 1:r
    p = P(t,:);
    rts = roots(p);
    rts = rts(rts==real(rts));
    L = length(rts);
    if L == 1
        rt(t) = rts;
    else
        funval = log(1+rts.^2)+rho.*(rts-s(t)).^2;
        rttem = rts(funval==min(funval));
        rt(t) = rttem(1);
    end
end
sigmaX = rt;
end