function err = ClusterErr(nClass,L,Idx)

NM = ClusterMatrix(nClass,L,Idx);

[assignMatrix,assignScore]=Hungary(-1*NM);
assignScore = -1*assignScore;

err = length(L)-assignScore;

err = err/length(L);

end