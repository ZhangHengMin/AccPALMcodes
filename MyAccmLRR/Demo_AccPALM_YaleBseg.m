clear all;
close all;
addpath('myLRR');
addpath(genpath('tools'));

choosedata = 'YaleB';
yaleBmat = zeros(2, 1);
pyaleBmat = zeros(2, 6); 
%---various lambda values-----
lamParas = [0.01 0.1 0.5 1.0 3.0 5.0 8.0 10.0] ;


for iparas = 1 : length(lamParas)
    lamdas = lamParas(iparas);
    disp([' lambda = ' num2str(lamdas)]);

    nClusNUM = [5 10];

    for nClus = 1 : length(nClusNUM)
        nCluster = nClusNUM(nClus);


        load YaleB               % load YaleB dataset
        num = nCluster * 64 ;    % number of data used for subspace segmentation
        X = fea(:,1:num) ;
        gnd = gnd(:,1:num) ;
        K = max(gnd);

        for i = 1 : size(X,2)
            X(:,i) = X(:,i) /norm(X(:,i)) ;
        end
        %
        ranknum = 10*nCluster;
        para.regType = 1;
        para.tol = 1e-5;
        para.mumax = 1e+6;
        para.maxIter = 200;
        para.decay = 0.95;
        %%

        para.maxR = ranknum;
        lambdaMax = topksvd(X, 1, 5);
        lambdaInit = para.decay*lambdaMax;
        lambda = lambdaInit;
        R = randn(size(X,2), para.maxR);  % having influences for the performance 
        [m, n] = size(X);
        %mu = lamdas*(m*n)^0.5;
        mu = lamdas*(max(m, n))^0.5;
        para.R = R;
        U0 = powerMethod(X, R, para.maxR, 1e-6 );
        para.U0 = U0;
        para.tau1 = 1.1*max(eig(X'*X));
        para.tau2 = 1.1;
        tic;
        [ Z, Y, err, Time, iter] = PowerALRR( X, mu, para, ranknum);
        time_cost = toc;

        
        %  
        for i = 1 : size(Z,2)
            Z(:,i) = Z(:,i) / max(abs(Z(:,i))) ;
        end
        Z = Selection( Z , nCluster ) ;
        Z = (Z + Z')/2; 
        idx = clu_ncut(Z,nCluster) ;
        acc = compacc(idx,gnd)*100; 

        disp(['nCluster num= ' num2str(nCluster), ' seg acc= ' num2str(acc), ' cost time= ' num2str(time_cost), ' iter num= ' num2str(iter)]);
    end 

end 
 





