function [ X, Y, err, Time, iter] = PowerALRR( O, mu, para, cluster )
% variables: X, Y, O
% model-par: lambda, mu
% algor-par
%
%
if(isfield(para, 'decay'))
    decay = para.decay;
else
    decay = 0.95;
end

if(isfield(para, 'maxR'))
    maxR = para.maxR;
else
    maxR = min(size(O));
end
%
regType = para.regType;
maxIter = para.maxIter;
mu_max = para.mumax;
tol = 1e-5;
tau1 = para.tau1;
tau2 = para.tau2;
% how to define?
L = eye(size(O,2)); Xpre = L;
S = zeros(size(O)); Ypre = S;


[~, Sv, V] = svd(para.U0'*O, 'econ');

V0 = V;
V1 = V;

% Acceleration Parameter
a0 = 1;
a1 = 1;
rho = 1.1;
%
lambdaMax = topksvd(O, 1, 10);
%err  = zeros(maxIter, 1);
obj0 = zeros(maxIter, 1);
obj1 = zeros(maxIter, 1);
%Time = zeros(maxIter, 1);
deltaObj = zeros(maxIter, 1);
totalTime = 0;
%
A = O; Y = O; iter = 0;
lambdaInit = decay*lambdaMax;
lambda = lambdaInit;
%for i = 1 : maxIter
while mu < mu_max
    timeFlag = tic;
    iter = iter+1;

    % update low rank
    Xo = L - (1/tau1)*A'*(A*L + S - O);
    [ R ] = filterBase(V1, V0, 1e-6);
    R = R(:,1:min(size(R,2), maxR));
    [Q, ~] = powerMethod(Xo, R, 3, 1e-5);
    hZ = Q'*Xo;
    [ U , Xsvs, V ] = myGSVT(hZ, lambda/(mu*tau1), regType);
    if(nnz(Xsvs) > 0)
        X = (Q*U)*(Xsvs*V');
        V0 = V1;
        V1 = V;
    end

    % update group sparse
    Yo = S - (1/tau2)*(A*L + S - O);
    Y = solve_l1l2(Yo, 1/(mu*tau2));

    % extragradients
    beta = (a0 - 1)/a1;
    Xeg = X + beta*(X - Xpre);
    Yeg = Y + beta*(Y - Ypre);
    % accleration technique
    ai = (1 + sqrt(1 + 4*a0^2))/2;
    a0 = a1;
    a1 = ai;

    %
    Xpre = X;
    Ypre = Y;
    SS  = randomized_svd(Xeg, cluster);
    %SS = svd(Xeg, 'econ');
    %XegSvS = Xsvs+beta*(Xsvs-Xsvs_pre);
    %Xsvs_pre = Xsvs;
    %Y00{iter} = X;
    % how to choose L and S
    err(iter) = norm(A*X + Y- O,'fro')/norm(O,'fro');
    obj0(iter) = getObjLRR(X, diag(Xsvs), Y, O, lambda, mu, regType);
    obj1(iter) = getObjLRR(Xeg, SS, Yeg, O, lambda, mu, regType);
    deltaObj(iter) = obj1(iter) - obj0(iter);

    if ( deltaObj(iter)  < 0)
        L = Xeg;  S = Yeg;
    else
        L = X;    S = Y;
    end
    % timing save
    totalTime = totalTime + toc(timeFlag);
    Time(iter) = totalTime;
    if(err(iter) < tol )
        break;
    end
    %Y0{iter} = X;
    % update mu
    mu =  rho*mu;
    lambda = decay*lambda;

end
err= err(1:iter);
Time= Time(1:iter);
obj0 = obj0(1:iter);
obj1 = obj1(1:iter);
end
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function  S = randomized_svd(A, k)
% Randomized SVD algorithm
% A - input matrix
% k - number of singular values/vectors to compute

% Get the size of A
[m, n] = size(A);

% Generate a random Gaussian matrix of size n x k
Omega0 = randn(n, k);

% Form the sample matrix Y = A * Omega0
EE = A * Omega0;

% Compute the QR factorization of Y, Q * R = Y
[Q, ~] = qr(EE, 0);

% Compute the product of A and Q, B = A' * Q
B = A' * Q;

% Compute the SVD of B, U * S * V' = B
S = svd(B, 'econ');

% Truncate S to the first k singular values
S = S(1:k);
end


function [E] = solve_l1l2(W,pp)
n = size(W,2);
E = W;
for i=1:n
    E(:,i) = solve_l2(W(:,i),pp);
end
%
    function [x] = solve_l2(w,pp)
        % min pp*|x|_2 + 1/2*|x-w|_2^2
        nw = norm(w);
        if nw > pp
            x = (nw-pp)*w/nw;
        else
            x = zeros(length(w),1);
        end


    end

end



 



