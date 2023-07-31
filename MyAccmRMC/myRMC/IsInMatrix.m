function y=IsInMatrix(a,A)
[m,n]=size(A); 
y=0; 
for i=1:m 
    for j=1:n 
        if A(i,j)==a 
            y=1; 
            break; 
        end 
    end 
    if j<n 
        break; 
    end 
end 