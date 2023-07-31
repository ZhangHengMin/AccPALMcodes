clear all;
close all;

addpath(genpath('tools')); 
addpath(genpath('myRMC')); 
addpath(genpath('1-Datasets'));
%
namelist_1 = {'jester-1','jester-2','jester-3','jester-all'};
Result_all = cell(1,4);
fid = 1;
% Jester Data
data_all = cell(1,4);
for dataset = 1 : 4
    fname = namelist_1{dataset};
    load(fname,'M');
    data_all{1,dataset} = M;
end
%% setup1

for rar = 1    :  2
    disp(['  rar = ' num2str( rar)]);

    alphaall = [1e-5 1e-4 0.001 0.005 0.01 0.1];
    for ialpha = 1 : length(alphaall)
        alpha = alphaall(ialpha);
        disp(['  alpha = ' num2str(alpha)]);


        beta = 0.95; % tuned


        %
        if rar == 1
            ratio = 0.5;  r = 30;
            load mask05;
            mask = mask05;
        else
            ratio = 0.8;  r = 30;
            load mask08;
            mask = mask08;
        end
  
        M = data_all{1,4};
        % Original mask and M
        mask1 = sign(sign(abs(M)-1e-3)+1);
        M_ori = M;
        [U, S, V] = svd(M_ori, 'econ');
        Svs = diag(S);
        M = U(:, 1:r)*diag(Svs(1:r))*V(:, 1:r)';
        [m,n] = size(M);
        mask_ori = mask1;
        Msub = M.*mask;


        %
        rankparas = [30]; %turnable
        for i = 1 : length(rankparas)
            Grank = rankparas(i);


            tol = 1e-5;
            param.decay = beta;    % tuned
            param.mumax = 1e+10;
            param.maxR = Grank;       % tuned
            lambdaMax = topksvd(Msub, 1, 10); %10
            lambdaInit = param.decay*lambdaMax;
            lambda = lambdaInit;
            %%
            R = randn(size(Msub,2), param.maxR);
            param.R = R;
            U0 = powerMethod(Msub, R, param.maxR, 1e-6 );
            param.U0 = U0;
            mu = alpha*(max(size(Msub)))^0.5; %
            %mu = alpha*(m*n)^0.5;
            tic;
            [M0, S, iter, obj, obj1, Time4] = PALMrmc_poweracc(Msub, Msub~=0, lambda, mu, param, tol);
            time_cost = toc;
 
            %%% Testing
            IndexM = (M - Msub)~=0;
            Rvec   = M0(IndexM) - M(IndexM);
            clear IndexM ii %Msub
            MAE = mean(abs(Rvec));
            NMAE = MAE/20;
            MSE = mean(Rvec.*Rvec);
            RMSE = sqrt(mean(Rvec.*Rvec));
            disp([' beta = ' num2str(beta), ' ratio = ' num2str(ratio), ' Grank = ' num2str(Grank), ' NMAE = ' num2str(NMAE), ' RMSE = ' num2str(RMSE), ' cost time= ' num2str(time_cost), ' number iter= ' num2str(iter)]);

        end




    end

end

  


