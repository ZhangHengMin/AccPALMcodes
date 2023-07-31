clear all; 
close all;
%clc;
%
addpath(genpath('1_datause'));
addpath(genpath('1-myRPCA'));
addpath(genpath('tools'));

% load('ExtYaleB_Sub3_38_12_96_84.mat');
% [a, b, c] = size(DAT);
% % for i = 1: size(Test_DAT,2)
% %     DS = reshape(Test_DAT(:,i), [96, 84]);
% %     Ds = imresize(DS, [32, 32]);
% %     D(:, i) = Ds(:);
% % end
% D = reshape(DAT,[a, b*c]);
% D = D(:,1:120);  imgsize = [96,84];


load('airport.mat');
imgsize=[size(X,1),size(X,2)];
D=im2double(reshape(X,[size(X,1)*size(X,2),size(X,3)]));
D = D(:,1:120); % 

%%
fid = 1;
para.regType = 1;
para.tol = 1e-5;
para.mumax = 1e+6;
para.maxIter = 200;

%% how to choose 
para.decay = 0.9; 
para.maxR = 20;
para.tau = 1.1;
% model paramerers
mu = 1000;
%
R = randn(size(D,2), para.maxR);
para.R = R;
U0 = powerMethod(D, R, para.maxR, 1e-6 );
para.U0 = U0;
%
for imageNuM = 1
    
    tic;
    [ X, Y, output ] = PowerARPCA( D, mu, para);
    %
    toc;
    plot(output.err, 'b', 'LineWidth',2);
    time_all=toc; 
    WE3 = [reshape(X(:,imageNuM),imgsize); reshape(Y(:,imageNuM),imgsize)];
    imshow(WE3);
    LS = X + Y;
    error = norm(LS(:)-D(:))/norm(D(:));
    fprintf('ours: Rank %d, Sparse Ratio %4.3f, errValue %4.2e%\n', rank(X),length(find(abs(Y)>0))/(size(D,1)*size(D,2)), error);
    
end

%     close all;
%     plot(out{1}.Time, out{1}.RMSE, 'b');
%     hold on;
%     plot(output.obj, 'r');
%     hold on;

%     plot(out{1}.Time, out{1}.obj, 'b');
%     hold on;
%     plot(out{2}.Time, out{2}.obj, 'r');
%     hold on;
%     close all;
%     plot(out{1}.RMSE, 'b');
%     hold on;
%     plot(out{2}.RMSE, 'r');
%     hold on;

%    legend('APGnc','AIRNN'); 
% figure(1);
% imshow([WE1, WE2, WE3, WE4,WE5,WE6, WE7, WE8,WE9]);
% p = ones(144*2, 1);
% z = cat(2, WE1, p, WE2, p, WE3, p, WE4, p, WE5, p, WE6, p, WE7, p, WE8, p, WE9);
% figure; imshow(z,[-1,1])

% figure(2);
% plot(log(errAPG.Total),'k-'); hold on;
% plot(log(errALM.Total),'k--'); hold on;
% plot(log(errPCP.Total),'c-'); hold on;
% plot(log(errPR.Total),'g-'); hold on;
% plot(log(errWR.Total),'m-'); hold on;
% plot(log(err23.Total),'r-'); hold on;
% plot(log(err12.Total),'b-'); hold on;
% plot(log(Aerr23.Total),'r--o'); hold on;
% plot(log(Aerr12.Total),'b--o'); hold on;
% legend('APG-RPCA','ALM-RPCA','PCP','PSSV-RPCA','WNNM-RPCA','PJIM(2/3)','PJIM(1/2)','APJIM(2/3)','APJIM(1/2)','Location','SouthWest');
% ylabel('Tre Values');
% xlabel('The Number of Iterations');



   