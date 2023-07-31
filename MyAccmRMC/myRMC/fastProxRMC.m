function [ X, Y, output ] = fastProxRMC( O, A, mu, theta1, theta2, para)

if(isfield(para, 'decay'))
    decay = para.decay;
else
    decay = 0.2;
end

if(isfield(para, 'maxR'))
    maxR = para.maxR;
else
    maxR = min(size(O));
end

tol = 1e-3;
tau1 = para.tau1;
tau2 = para.tau2;
regType = para.regType;
maxIter = para.maxIter;

Y = zeros(size(O));
X = O;
idx = find(A==1);
idx1 = find(A~=1);
V = randn(size(O,2), 1);
V = powerMethod(O, V, 1, 1e-6);
[~, ~, V] = svd(V'*O, 'econ');
V0 = V;
V1 = V;

lambdaMax = topksvd(O, 1, 5);
lambdaInit = 0.9*lambdaMax;
lambda = lambdaInit;
obj  = zeros(maxIter, 1);
Time = zeros(maxIter, 1);
PSNR = zeros(maxIter, 1);
totalTime = 0;
for i = 1:maxIter
    timeFlag = tic;
    
    % setup loop parameter
    switch(regType)
        case 1 % CAP
            lambdai = abs(lambdaMax - lambda)*(decay^i) + lambda/tau;
            theta1i = theta1 + (decay^i)*lambdai;
        case 2 % Logrithm
            lambdai = lambda;
            theta1i = theta1;
        case 3 % TNN
            lambdai = abs(lambdaMax - lambda)*(decay^i) + lambda;
            theta1i = theta1;
        otherwise
            assert(false);
    end

    X = X - (1/tau1)*(X + Y - O);
    [ R ] = filterBase( V1, V0, 1e-6);
    R = R(:,1:min(size(R,2), maxR));
    [Q, pwIter] = powerMethod( X, R, 3, 1e-5);
    hZ = Q'*X;
    [ U , S, V ] = myGSVT(hZ, lambda/(mu*tau1), 1);
%     [ U, S, V ] = GSVT( hZ, lambdai, theta1i, regType);
    if(nnz(S) > 0)
        X = (Q*U)*(S*V');
        V0 = V1;
        V1 = V;
    end
    
    Yo = Y - (1/tau2)*(X + Y - O);
%     Y = reshape(Y, numel(Y), 1);
%     Y = proximalRegC(Y, length(Y), mu/tau, theta2, 1);
%     Y = reshape(Y, size(X,1), size(X,2));
    % update group sparse
%    Yo = S - (1/tau2)*A.*(X + Y - O);
%     for ii = 1:length(idx)
%         Y(idx(ii)) = findrootp(Yo(idx(ii)), 1/(mu*tau2), 1);
%     end
%     Y(idx1) = Yo(idx1);
   Half = 2;
    if Half == 1
        Y(A) = ST12(Yo(A), 2/(mu*tau2));         % Half-Thresholding operator
    elseif Half == 2
        Y(A) = solve_Lp(Yo(A), 1/(mu*tau2), 0.8);  % Generalized Soft-Thresholding
    end
    
    totalTime = totalTime + toc(timeFlag);
    Time(i) = totalTime;
    obj(i) = getObjRMC(X, S, Y, O, A, lambda, mu, 1);
%    obj(i) = getObjRPCA(X, S, Y, O, lambda, theta1, regType, mu, theta2);   
    if(i == 1)
        deltaObj = inf;
    else
         deltaObj = obj(i - 1) - obj(i);
    end

%     fprintf('iter %d, obj %.4d(dif %.2d), rank %d, lambda %.2d, power(rank %d, iter %d), nnz:%0.2f \n', ...
%             i, obj(i), deltaObj, nnz(S), lambdai, size(R, 2), pwIter, nnz(Y)/numel(Y));
        
%     if(isfield(para, 'test'))
%         PSNR(i) = psnr(X + Y, para.test, 1);
%         fprintf('PSNR %.3d \n', PSNR(i));
%     end

    if(abs(deltaObj) < tol)
        break;
    end
end

output.S    = diag(S);
output.rank =  nnz(S);

output.obj  = obj(1:i);
output.Time = Time(1:i);
output.PSNR = PSNR(1:i);

 

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

