clc
N=25;
format shortG
format loose
y=importdata("results/y.dat");
Sigma=importdata("results/Sigma.dat");
Imp_Data=importdata("files/mainData.dat");
a=Imp_Data(1,1);
b=Imp_Data(1,2);
pa=Imp_Data(2,1);
pb=Imp_Data(2,2);
E=Imp_Data(3,1);
nyu=Imp_Data(3,2);
uk=Imp_Data(4,1);
rk=Imp_Data(4,2);

r=linspace(a,b,N);
plot(r,Sigma(:,1))
y_an=@(r)(((1-2*nyu)*(1+nyu)/E)*((pa*a^2-pb*b^2)/(b^2-a^2))*r+((1+nyu)/E)*((a^2+b^2)/r)*...
    ((pa-pb)/(b^2-a^2)))*uk;
plot(r,y_an(r))