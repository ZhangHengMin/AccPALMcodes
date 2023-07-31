clear all;
close all;
addpath('myLRR');
addpath('0SpCluster');
addpath(genpath('hopkins155'));
addpath(genpath('tools'));

%%
filename = 'hopkins155';
data = load_motion_data(1);
data(140)=[];

%
fid = 1;
maxNumGroup = 5;
num = zeros(maxNumGroup,1);
for run = 1 
allranknum = [6];
for ii = 1 : length(allranknum)
    %% how to choose
    ranknum = allranknum(ii);
    for t = 1 : length(data)
        
        X = data(t).X;
        gnd = data(t).ids;
        K = max(gnd);
        
        para.regType = 1;
        para.tol = 1e-6;
        para.mumax = 1e+6;
        para.maxIter = 200;
        %%
        
        para.maxR = ranknum;
        para.tau1 = 3.5*max(eig(X'*X))+0.1;
        para.tau2 = 2.5;
        
        R = randn(size(X,2), para.maxR);
        [m, n] = size(X);
        mu = 0.20*(m*n)^0.5;
        para.decay = 0.9;
        para.R = R;
        U0 = powerMethod(X, R, para.maxR, 1e-6 );
        para.U0 = U0;
        tic;
        [ Z, Y, output,iter] = PowerALRR( X, mu, para );
        time = toc;
        
        for i = 1 : size(Z,2)
            Z(:,i) = Z(:,i) / max(abs(Z(:,i))) ;
        end
        %
        Z = ( abs(Z) + abs(Z') ) / 2 ;
        %imagesc(Z); colorbar; %impixelinfo;
        %post processing
        [U,S,V] = svd(Z,'econ');
        S = diag(S);
        r = sum(S>1e-4*S(1));
        U = U(:,1:r);S = S(1:r);
        U = U*diag(sqrt(S));
        U = normr(U);
        L = (U*U').^6;
        % spectral clustering
        D = diag(1./sqrt(sum(L,2)));
        L = D*L*D;
        [U,S,V] = svd(L);
        V = U(:,1:K);
        V = D*V;
        %
        rand('state',0);
        idx = kmeans(V,K,'emptyaction','singleton','replicates',20,'display','off');
        %err0 =  missclassGroups(idx,gnd,K)/length(idx);
        err0 = 1 - compacc(idx,gnd);
        
        if t == 10|| t == 50 || t == 100 || t == 150
            disp(['seq ' num2str(t) ',err=' num2str(err0) ',err=' num2str(K)]);
        end
        num(K) = num(K)+1;
        allrateZ{K}(num(K)) = err0*100;
        alltimeZ{K}(num(K)) = time;
        time23{t} = output.Time;
        obj{t} = output.obj;
    end
    
    
    
    % results
    L = [2 3];
    for i = 1:length(L)
        j = L(i);
        avgrateZ(j) = mean(allrateZ{j});
        medrateZ(j) = median(allrateZ{j});
        sumtime(j) = mean(alltimeZ{j});
    end
    avgallZ = mean([allrateZ{2} allrateZ{3}]);
    sumtimeZ = mean([alltimeZ{2} alltimeZ{3}]);
    medallZ = median([allrateZ{2} allrateZ{3}]);
    
    disp(['  two motions, mean = ' num2str(avgrateZ(2)) ', median = ' num2str(medrateZ(2)) ', time = ' num2str(sumtime(2))])
    disp(['three motions, mean = ' num2str(avgrateZ(3)) ', median = ' num2str(medrateZ(3)) ', time = ' num2str(sumtime(3))])
    disp(['  all motions, mean = ' num2str(avgallZ) ', median = ' num2str(medallZ) ', time = ' num2str(sumtimeZ)])
    
%     Cluster2Err(ii) = avgrateZ(2);
%     Cluster3Err(ii) = avgrateZ(3);
%     ClusterALLErr(ii) = avgallZ;
end
end
% figure;
% a1 = plot(time23{4},log(obj{4}),'-g','LineWidth',2.5); hold on;
% a2 = plot(time23{6},log(obj{6}),'-b','LineWidth',2.5); hold on;
% 
% ylabel(' Values of Objective Function ($log$)','interpreter','latex', 'FontSize',12);
% %xlabel(' Number of Iteration','interpreter','latex', 'FontSize',12);
% xlabel(' CPU Time (seconds)','interpreter','latex', 'FontSize',12);
% legend([a1 a2],'2 Motions','3 Motions','Location','northeast')


% % This Function to Estimate the Rank of the Input Matrix
% function d = rank_estimation(X)
% %
% [n, m]   = size(X);
% epsilon = nnz(X)/sqrt(m*n);
% mm = min(100, min(m, n));
% S0 = lansvd(X, mm, 'L');
% 
% S1  = S0(1:end-1)-S0(2:end);
% S1_ = S1./mean(S1(end-10:end));
% r1  = 0;
% lam = 0.05;
% while(r1 <= 0)
%     for idx = 1:length(S1_)
%         cost(idx) = lam*max(S1_(idx:end)) + idx;
%     end
%     [v2, i2] = min(cost);
%     r1 = max(i2-1);
%     lam = lam + 0.05;
% end
% clear cost;
% 
% for idx = 1:length(S0)-1
%     cost(idx) = (S0(idx+1)+sqrt(idx*epsilon)*S0(1)/epsilon )/S0(idx);
% end
% [v2, i2] = min(cost);
% r2 = max(i2);
% d = max([r1 r2]);
% end

% figure;
% x = 1:9;
% Cluster2Err = [4.890 4.069 4.303 1.667 2.257 3.019 3.508 4.045 4.724];
% Cluster3Err = [8.717 7.103 5.947 4.623 4.952 4.990 4.919 5.00 5.446];
% ClusterALLErr = [5.754 4.754 4.674 2.334 2.865 3.464 3.827 4.261 4.887   ];
% %
% a1 = plot(x,Cluster2Err,'^-g','LineWidth',2.5); hold on;
% a2 = plot(x,Cluster3Err,'o-b','LineWidth',2.5); hold on;
% a3 = plot(x,ClusterALLErr,'*-k','LineWidth',2.5); hold off;
% 
% %title(' 10-Training and 50-Testing')
% ylabel(' Values of Mean Error (\%)','interpreter','latex', 'FontSize',12);
% xlabel(' Number of Rank Estimation','interpreter','latex', 'FontSize',12);
% legend([a1 a2 a3],'2 Motions','3 Motions', 'All Motions','Location','northeast')
% set(gca,'xticklabel',{'3','4','5','6','7','8','9','10','11'});
% %set(gca,'xticklabel',{'2','3'});

