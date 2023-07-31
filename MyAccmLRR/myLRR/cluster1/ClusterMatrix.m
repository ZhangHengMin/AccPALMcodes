function NM = ClusterMatrix(nClass,L,Idx)
for s = 1:nClass
    nn = (L==s);
    for t = 1:nClass
        mm = (Idx==t);
        nn = reshape(nn,length(nn),1);
        mm = reshape(mm,length(mm),1);
        NM(s,t) = sum((nn.*mm)==1);
    end
end
end
