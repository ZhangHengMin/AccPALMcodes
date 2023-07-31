function [ X,S,B,funval ] = Subspace_XSolver(A,alpha,beta,gamma,maxiter,toler)

% SCLA: 
% \argmin_{X,S,B} logdet(I + X^TX) + \alpha\|S\|_{l}  
%                 + \beta\|A - B - S\|_{F}^{2} + \gamma\|B - BX\|_{F}^{2}
% Input
%     A is the p-by-n data matrix;
%
% Output
%     X is the representation matrix
%     S is the sparse matrix
%     B is the underlying clean matrix
%
% If you use our code, please cite the following paper:
%
% Peng, Chong, Zhao Kang, Huiqing Li, and Qiang Cheng. 
% "Subspace Clustering Using Log-determinant Rank Approximation."
% In Proceedings of the 21th ACM SIGKDD International Conference 
% on Knowledge Discovery and Data Mining,
% pp. 925-934. ACM, 2015.
% 
% We appreciate it if you would also cite:
% 
% Kang, Zhao, Chong Peng, Jie Cheng, and Qiang Cheng. 
% "Logdet rank minimization with application to subspace clustering."
% Computational intelligence and neuroscience 2015 (2015).
% 
% Kang, Zhao, Chong Peng, and Qiang Cheng. 
% "Robust Subspace Clustering via Smoothed Rank Approximation." (2015).
% 
% We provide this code for research purpose use only. 
% However, we do not have any guarantee for using this code.
% If you have any question using this code, please contact pchong@siu.edu


[m,n] = size(A);
X0 = eye(n);
Y0 = eye(n) -X0;
S0 = zeros(m,n);
B0 = A;
Gamma = zeros(n);
rho = 1;
delta = 1.1;

for t = 1:maxiter
    
    D = eye(n)-Y0-Gamma/rho;
    X  = SCLA_Logdet2_X(D,rho/2);
    
    BB = gamma*(Y0*Y0')+beta*eye(n);
    B = beta*(A-S0)*BB^(-1);
    
    S = S_Solver_L21(A-B,alpha/beta/2);
    %     S = S_Solver_L1(A-B,alpha/beta/2);
     
    funval(t) = log(det(eye(n)+X'*X))+alpha*sum(sum(abs(S)))+beta*sum(sum((A-B-S).^2))+gamma*sum(sum((B-B*X).^2));
    
    YY = 2*gamma*(B'*B)+rho*eye(n);
    Y = YY^(-1)*rho*(eye(n)-X-Gamma/rho);
    
    Gamma = Gamma+rho*(Y-eye(n)+X);
    rho = rho*delta;
    
    err = max([sum(sum((X-X0).^2)),sum(sum((B-B0).^2)),sum(sum((S-S0).^2))]);
    
    if err <= toler
        break;
    end
    
    X0 = X; B0 = B; S0 = S; Y0 = Y;
    
end


end

function [ X ] = SCLA_Logdet2_X(D,rho)

[U,S,V] = svd(D);
S0 = diag(S);
r = length(S0);

P = [rho*ones(r,1), -1*rho*S0, (rho+1)*ones(r,1), -1*rho*S0];

rt = zeros(r,1);

for t = 1:r
    p = P(t,:);
    rts = roots(p);
    
    rts = rts(rts==real(rts));
    
    L = length(rts);
    if L == 1
        rt(t) = rts;
    else
        funval = log(1+rts.^2)+rho.*(rts-S0(t)).^2;
        rttem = rts(funval==min(funval));
        rt(t) = rttem(1);
    end
end

% sig = diag(rt);
% X = U*sig*V';

SSS = diag(rt);
[m,n] = size(D);
sig = zeros(m,n);
sig(1:min(m,n),1:min(m,n)) = SSS;
X = U*sig*V';

end

function G3 = S_Solver_L21(G,lambda)
G1 = sqrt(sum(G.^2,1));
G1(G1==0) = lambda;
G2 = (G1-lambda)./G1;
G3 = G*diag((G1>lambda).*G2);

end

function [ S ] = S_Solver_L1(D,lambda)

S = zeros(size(D));
DD = abs(D)-lambda;
DD2 = DD.*sign(D);
ID = (DD>0);
S(ID) = DD2(ID);

end
