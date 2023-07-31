function [ X, Y, iter, rankvalue, err, Time] = AccPALM_LRR(O, mu, para, tol)
% variables: X, Y, O
% model-par: lambda, mu
% algor-par: decay, regType, mumax, tau1, tau2
%
% O: data matrix with missing entries
% A: index set of observed entries in O
% mu: penalty parameter for penalty function
% para: involved parameters
% tol: tolerance level for convergence
% X: low-rank matrix
% Y: sparse error matrix
% iter: number of iterations
% rankvalue: rank of low-rank matrix X
% err: relative error between (X+Y) and observed entries in O
% Time: time taken for each iteration of algorithm

% Set default parameters
if(isfield(para, 'decay'))
    decay = para.decay;
else
    decay = 0.9;
end

if(isfield(para, 'regType'))
    regType = para.regType;
else
    regType = 1;
end

 
mu_max = 1000;
% Initialize variables
L = eye(size(O,2)); 
Xpre = L;
S = zeros(size(O));
Ypre = S;
Y = O;
A = O;



% Acceleration parameters
a0 = 1;
a1 = 1;
rho = 1.1;
lambdaMax = topksvd(O, 1, 10);
maxR = para.maxR;
V = randn(size(O,2), 1);
V = powerMethod(O, V, 1, 1e-6);
[~, ~, V] = svd(V'*O, 'econ');
V0 = V;
V1 = V;


% Lipschitz constants
tau1 = 3.5*max(eig(O'*O))+0.1;
tau2 = 2.5;
lambdaInit = decay*lambdaMax;
lambda = lambdaInit;
iter = 0;
totalTime = 0;
while mu < mu_max
    timeFlag = tic;
    iter = iter+1;


    % Update low-rank matrix
    Xmed = L - (1/tau1)*A'*(A*L + S - O);
    %     [U, Xsvs, V] = myGSVT(Xmed, lambda/(mu*tau1), regType);
    %     X = U*Xsvs*V';
    R  = filterBase(V1, V0, 1e-6);
    R = R(:,1:min(size(R,2), maxR));
    [Q1, ~] = powerMethod(Xmed, R, 3, 1e-5);
    hZ = Q1'*Xmed;
    [ U , Sg, V ] = myGSVT(hZ, lambda/(mu*tau1), regType);
    if (nnz(Sg) > 0) || iter == 1
        X = (Q1*U)*(Sg*V');
        V0 = V1;
        V1 = V;
    end


    % Update sparse error matrix
    Ymed = S - (1/tau2)*(A*L + S - O);
    Y = solve_l1l2(Ymed, 1/(mu*tau2));

    % Compute extragradient
    beta = (a0 - 1)/a1;
    Xeg = X + beta*(X - Xpre);
    Yeg = Y + beta*(Y - Ypre);

    % Compute Nesterov's acceleration
    t = (1 + sqrt(1 + 4 * a1^2)) / 2;
    a0 = a1;
    a1 = t;
    XegSvS  = randomized_svd(Xeg, rr);
    XegSvS = svd(Xeg, 'econ');

    % Compute objective function and error value
    obj(iter) = getObjLRR(X, diag(Sg), Y, O, lambda, mu, regType);
    obj1(iter) = getObjLRR(Xeg, XegSvS, Yeg, O, lambda, mu, regType);

    %obj(iter) = getObjRMC(X, Sg, Y, O, Omega, lambda, mu, regType);
    %XegSvS  = randomized_svd(Xeg, rr);
    %obj1(iter) = getObjRMC(Xeg, diag(XegSvS), Yeg, O, Omega, lambda, mu, regType);
    deltaObj(iter) = obj1(iter) - obj(iter);

    % Update low-rank and sparse error matrices
    if (deltaObj(iter) < 0)
        L = Xeg;
        S = Yeg;
    else
        L = X;
        S = Y;
    end
    %err(iter) = max(norm(X - Xpre, 'fro'), norm(E - Epre, 'fro'));
    err(iter) = norm(A*X + Y- O,'fro')/norm(O,'fro');
    % Timing save
    totalTime = totalTime + toc(timeFlag);
    Time(iter) = totalTime;

    % Check stopping criterion
    if(err(iter) < tol)
        break;
    end

    % Update acceleration parameters and regularization parameter lambda

    mu = rho*mu;
    %lambda = decay*lambda;

    % Update previous variables
    Xpre = X;
    Ypre = Y;
end

rankvalue = rank(X);

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




