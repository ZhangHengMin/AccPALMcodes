clear all;
close all;


%%
addpath(genpath('2-compared'));
addpath(genpath('IRNN'));
Xfull = double(imread('testimage.jpg'));
[m,n,c]=size(Xfull);

mask(:,:,1)=ind;
mask(:,:,2)=ind;
mask(:,:,3)=ind;
Xmiss = Xfull.*mask;
 
%% % choose penalty in IRNN
fun = 'lp' ;        gamma = 0.5;
% fun = 'scad' ;      gamma = 10;
% fun = 'logarithm' ; gamma = 0.1; % or 1
% fun = 'mcp' ;       gamma = 0.1;
% fun = 'etp' ;       gamma = 0.001;

lambda_rho = 0.5;
tic;
for i = 1 : 3
    fprintf('chanel %d\n',i) ;
    X = Xfull(:,:,i);
    x = X(:) ;
    y = M(x,1) ;   
    lambda_Init = max(abs(M(y,2)))*1000;
    Xhat(:,:,i) = IRNN(fun,y,M,m,n,gamma,lambda_Init,lambda_rho);
end
time_cost = toc;
Xhat = max(Xhat,0);
Xhat = min(Xhat,255);
psnrIRNN = PSNR(Xfull,Xhat,max(Xfull(:)));
disp([' IRNN_PSNR = ' num2str(psnrIRNN), ' IRNNtime= ' num2str(time_cost)]);
figure(1)
subplot(1,2,1)
imshow(Xmiss/255);
subplot(1,2,2)
imshow(Xhat/255);




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic;
for ch = 1 : 3
    fprintf('chanel %d\n',ch) ;
    [L, obj] = LpRtracep_new(Xmiss(:,:,ch), ind, 1.0, 0.5, 0.5);
    Xhat(:,:,ch) = L; 
end
time_cost = toc;
Xhat = max(Xhat,0);
Xhat = min(Xhat,255);
psnrSpLq = PSNR(Xfull,Xhat,max(Xfull(:)));
disp([' SpLq_PSNR = ' num2str(psnrSpLq), ' SpLqtime= ' num2str(time_cost)]);


figure(2)
subplot(1,2,1)
imshow(Xmiss/255);
subplot(1,2,2)
imshow(Xhat/255);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

param.TOL =1e-5;
param.MAX_ITER = 5e2;
param.MAX_RANK = round(0.25*min(m,n));
lamParas =  [0.01 0.1 1.0 5.0 10.0];
for iparas = 1 : length(lamParas)
    lamdas = lamParas(iparas);
    %disp([' lambda = ' num2str(lamdas)]);

    tic;
    for ch=1:3 
        [L, S, iter] = NRMC_BCD(Xmiss(:,:,ch), ind, lamdas, 0.5, 0.5, param);
        disp([' chanel = ' num2str(ch), ' iternum = ' num2str(iter)]);
        Xhat(:,:,ch) = L; 
    end
    time_cost = toc;
    Xhat = max(Xhat,0);
    Xhat = min(Xhat,255);
    ourpsnr = PSNR(Xfull,Xhat,max(Xfull(:)));
    disp([' lambda = ' num2str(lamdas), ' iternum = ' num2str(iter),' ProposedPSNR = ' num2str(ourpsnr), ' time= ' num2str(time_cost)]);
    figure(3)
    subplot(1,2,1)
    imshow(Xmiss/255);
    subplot(1,2,2)
    imshow(Xhat/255);

end
