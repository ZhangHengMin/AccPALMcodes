function [ X, Y, output, i ] = PowerARMC1(O, A, mu, para )
% variables: X, Y, O
% model-par: lambda, mu
% algor-par
%
%
if(isfield(para, 'decay'))
    decay = para.decay;
else
    decay = 0.9;
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

idx = find(A==1);
idx1 = find(A~=1);
%
% Xsvs_pre = svd(L,'econ');
% Xsvs_pre = diag(Xsvs_pre);
% V = randn(size(O,2), 1);
% V = powerMethod(O, V, 1, 1e-6);
[~, Sv, V] = svd(para.U0'*O, 'econ');
Xsvs_pre = Sv;
V0 = V;
V1 = V;

% Acceleration Parameter
a0 = 1;
a1 = 1;
rho = 1.05;
%
lambdaMax = topksvd(O, 1, 10);
err  = zeros(maxIter, 1);
obj0 = zeros(maxIter, 1);
Time = zeros(maxIter, 1);
deltaObj = zeros(maxIter, 1);
totalTime = 0;
%
Y = O; i = 0;
lambdaInit = decay*lambdaMax;
lambda = lambdaInit;
% lambda = lambda_Init;
lambda_Target = lambdaInit * 1e-5;
%while lambda > lambda_Target
%for i = 1 : 50
while mu < mu_max
    timeFlag = tic;
    i = i+1;
    
    % update low rank
    X = L - (1/tau1)*A.*(L + Y - O);
    [ R ] = filterBase(V1, V0, 1e-6);
    R = R(:,1:min(size(R,2), maxR));
    [Q, ~] = powerMethod(X, R, 3, 1e-5);
    hZ = Q'*X;
    [ U , Xsvs, V ] = myGSVT(hZ, lambda/(mu*tau1), regType);
    if(nnz(Xsvs) > 0)
        X = (Q*U)*(Xsvs*V');
        V0 = V1;
        V1 = V;
    end
    % update group sparse
    Yo = S - (1/tau2)*A.*(X + S - O);
%     for ii = 1:length(idx)
%         Y(idx(ii)) = findrootp(Yo(idx(ii)), 1/(mu*tau2), 1);
%     end
     Y(idx1) = Yo(idx1);
     Y(idx) = sign(Yo(idx)).*max(Yo(idx)-1/(mu*tau2),0);
    
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
    XegSvS = Xsvs+beta*(Xsvs-Xsvs_pre);
    Xsvs_pre = Xsvs;
    
    % how to choose L and S
    err(i) = norm(A.*(X + Y- O),'fro')/norm(O,'fro');
    obj0(i) = getObjRMC(X, Xsvs, Y, O, A, lambda, mu, regType);
    obj1(i) = getObjRMC(Xeg, XegSvS, Yeg, O, A, lambda, mu, regType);
    deltaObj(i) = obj1(i) - obj0(i);
    
    if ( deltaObj(i)  < 0)
        L = Xeg;  S = Yeg;
    else
        L = X;    S = Y;
    end
    % timing save
    totalTime = totalTime + toc(timeFlag);
    Time(i) = totalTime;
    if(err(i) < tol )
        break;
    end
    % update mu
    mu =  rho*mu;
    
%     lambda = 0.9*lambda;
    
    %end
    
    %output.S    = diag(Xsvs);
    output.rank =  nnz(Xsvs);
    output.Time = Time(1:i);
    output.err = err(1:i);
    output.obj  = obj0(1:i);
    
end

% min_{x}  0.5*(x-a)^2 + r*|x|^p
function x = findrootp(a, r, p)

x = 0;
if p == 1  
    if a > r
        x = a-r;
    elseif a < -r
        x = a+r;
    end;
elseif p == 0
    if a > sqrt(2*r) || a < -sqrt(2*r)
        x = a;
    end;
else
v = (r*p*(1-p))^(1/(2-p))+eps;
v1 = v+r*p*v^(p-1);
ob0 = 0.5*a^2;
if a > v1
    x = a;
    for i = 1:10
        f = (x-a) + r*p*x^(p-1);
        g = 1-r*p*(1-p)*x^(p-2);
        x = x-f/g;
    end;
    ob1 = 0.5*(x-a)^2 + r*x^p;
    x_can = [0,x];
    [temp,idx] = min([ob0,ob1]);
    x = x_can(idx);
elseif a < -v1
    x = a;
    for i = 1:10
        f = (x-a) - r*p*abs(x)^(p-1);
        g = 1-r*p*(1-p)*abs(x)^(p-2);
        x = x-f/g;
    end;
    ob1 = 0.5*(x-a)^2 + r*abs(x)^p;
    x_can = [0,x];
    [temp,idx] = min([ob0,ob1]);
    x = x_can(idx);
end;
end;
1;

