function [NMAE, RMSE] = evaluateCriteria(M, Msub, M0)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    IndexM = (M - Msub)~=0;
    Rvec   = M0(IndexM) - M(IndexM);
    clear IndexM ii %Msub
    MAE = mean(abs(Rvec));
    NMAE = MAE/20; 
    MSE = mean(Rvec.*Rvec);
    RMSE = sqrt(mean(Rvec.*Rvec))/20;
end

