
function [y,fval]=Hungary(C)
[m,n]=size(C); 
tempC=C; 

for i=1:m 
    tempC(i,:)=tempC(i,:)-min(tempC(i,:)); 
end 
for i=1:n 
    tempC(:,i)=tempC(:,i)-min(tempC(:,i)); 
end        
tempC=TryAssign(tempC);               
OneNumber=0;
for i=1:m                         
    for j=1:n 
        if tempC(i,j)==inf 
            OneNumber=OneNumber+1; 
            break; 
        end 
    end 
end 
while OneNumber<m
    Row=zeros(m,1);      
    Col=ones(1,n);       
    Line=[]; 
    for i=1:m 
        if IsInMatrix(inf,tempC(i,:))==0 
            Line=[Line,i]; 
            Row(i)=1; 
        end 
    end 
    for i=Line 
        Cur=i; 
        while Cur~=0 
            [Cur,Row,Col]=ZeroCover(tempC,Row,Col,Cur);
        end 
    end 
    temp=inf; 
    for i=1:m      
        for j=1:n 
            if Row(i)==1&Col(j)==1&tempC(i,j)<temp 
                temp=tempC(i,j); 
            end 
        end 
    end 
    for i=1:m 
        for j=1:n 
            if tempC(i,j)==inf|tempC(i,j)==-inf 
                tempC(i,j)=0; 
            end 
        end 
    end 
    for i=1:m    
        if Row(i)==1 
            tempC(i,:)=tempC(i,:)-temp; 
        end 
    end 
    for j=1:n    
        if Col(j)==0 
            tempC(:,j)=tempC(:,j)+temp; 
        end 
    end 
    tempC=TryAssign(tempC); 
    OneNumber=0;
    for i=1:m                         
        for j=1:n 
            if tempC(i,j)==inf 
                OneNumber=OneNumber+1; 
                break; 
            end 
        end 
    end 
end 
AssignMatrix=zeros(m,n);
for i=1:m 
    for j=1:n 
        if tempC(i,j)==inf 
            AssignMatrix(i,j)=1; 
        end 
    end 
end 
 y=AssignMatrix; 
temp=C.*AssignMatrix; 
fval=sum(temp(:)); 