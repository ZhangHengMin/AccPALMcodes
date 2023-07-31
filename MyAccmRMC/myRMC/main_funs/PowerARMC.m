function [ X, Y,i ] = PowerARMC(O, A, mu, para )
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
tol = 1e-6;
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
%V = powerMethod(O, V, 1, 1e-6);
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
Y = ones(size(O)); i = 0;
lambdaInit = decay*lambdaMax;
lambda = lambdaInit;
% lambda = lambda_Init;
% lambda_Target = lambdaInit * 1e-5;
% while lambda > lambda_Target
%     for i = 1 : 50
while mu < mu_max
    timeFlag = tic;
    i = i+1;
    
    % update low rank
    Xo = L - (1/tau1)*(L + Y - O);
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
    Yo = S - (1/tau2)*(X + S - O);
    Half = 2;
    if Half == 1
        Y(A) = ST12(Yo(A), 2/(mu*tau2));         % Half-Thresholding operator
    elseif Half == 2
        Y(A) = solve_Lp(Yo(A), 1/(mu*tau2), 0.8);  % Generalized Soft-Thresholding
    end
    
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
    
    %      lambda = 0.9*lambda;
    
    %end
    
%     %output.S    = diag(Xsvs);
%     output.rank =  nnz(Xsvs);
     output.Time = Time(1:i);
%     output.err = err(1:i);
%     output.obj  = obj0(1:i);
    
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

% The generalized soft-thresholding operator
function  x = solve_Lp( y, lambda, p )

% Modified by Dr. Weisheng Dong 
J     =   3;
tau   =  (2*lambda.*(1-p))^(1/(2-p)) + p*lambda.*(2*(1-p)*lambda)^((p-1)/(2-p));
x     =   zeros( size(y) );
i0    =   find( abs(y) > tau );

if length(i0) >= 1
    y0    =   y(i0);
    t     =   abs(y0);
    for  j  =  1 : J
        t    =  abs(y0) - p*lambda.*(t).^(p-1);
    end
    x(i0)   =  sign(y0).*t;
end


% The half-thresholding operator
function w = ST12(temp_v, gamma)
   
temp_c = 54^(1/3)*(gamma)^(2/3)/4;
temp_w = abs(temp_v) > temp_c; 
%temp_H = temp_w.*(abs(temp_v/3).^(-3/2)); 
temp_H = acos((gamma/8)*(temp_w.*(abs(temp_v/3).^(-3/2))));
temp_H = temp_w.*(1 + cos((2/3)*pi - (2/3)*temp_H));
w = (2/3)*temp_v.*temp_H;
