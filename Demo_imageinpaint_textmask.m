clear all;
close all;


% load data 
addpath(genpath('MyAccmRMC'));

% ---nonstrictly low-rank-----
Xfull = double(imread('demo_figure.jpg'));
[m, n, chnum]=size(Xfull);

% ---strictly low-rank-----
r = 50;
for ch = 1 :3
    [S,V,D] = svd(Xfull(:,:,ch));
    v = diag(V);
    v(r+1:end) = 0;
    Xfull(:,:,ch) = S*diag(v)*D';
end



ind = im2bw(imread('mask11.jpg'));   % text

mask(:,:,1)=ind;
mask(:,:,2)=ind;
mask(:,:,3)=ind;
Xmiss = Xfull.*mask;  % obseved data

%---various method names-----
tol = 1e-5; 
lamdas = 1e-4;  % tuned
param.regType = 1; 
param.decay = 0.95;    % tuned
param.mumax = 1e+10;

for ch = 1 : 3


    param.maxR = r;
    lambdaMax = topksvd(Xmiss(:,:,ch), 1, 5); %10
    lambdaInit = param.decay*lambdaMax;
    lambda = lambdaInit;
    %%
    R = randn(size(Xmiss(:,:,ch),2), param.maxR);
    param.R = R;
    U0 = powerMethod(Xmiss(:,:,ch), R, param.maxR, 1e-6 );
    param.U0 = U0;
    mu = lamdas*(max(m, n))^0.5;
    tic;
    [L, S, iter, rankvalues, obj, obj1, Time] = PALMrmc_poweracc(Xmiss(:,:,ch), ind, lambda, mu, param, tol);

    timech(ch) = toc;
    iterch(ch) = iter;
 
    disp([' chanel = ' num2str(ch), ' iternum = ' num2str(iter), ' ranknum = ' num2str(rankvalues)]);
    Xhat(:,:,ch) = L;
end
% Note that the time is longer because of the computations of objective
% function
time_cost = sum(timech);
iter_cost = sum(iterch);
Xhat = max(Xhat,0);
Xhat = min(Xhat,255);
% Results
PSNR_value = PSNR(Xfull,Xhat,max(Xfull(:)));
SSIM_value = ssim(Xfull,Xhat);
disp([' decay = ' num2str(param.decay), ' psnr= ' num2str(PSNR_value), ...
    ' SSIM_value = ' num2str(SSIM_value), ' cost time= ' num2str(time_cost), ' cost iter= ' num2str(iter_cost)]);

figure;
subplot(1,2,1)
imshow(Xmiss/255);
subplot(1,2,2)
imshow(Xhat/255);