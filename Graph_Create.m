clc
format shortG
format loose
sep="_";
NN="200";
route=@(Dim,Type,obj,size,AS)"results/"+Dim+Type+obj+sep+size+sep+AS+".dat";
Imp_Data=importdata("files/mainData.dat");
y=importdata(route("1D/","Non_Schwarz/","y",NN,"1"));
Sigma_NS=importdata(route("1D/","Non_Schwarz/","Sigma",NN,"1"));
Sigma_S1=importdata(route("1D/","Schwarz/","Sigma",NN,"2"));
Sigma_S2=importdata(route("1D/","Schwarz/","Sigma",NN,"4"));
Sigma_S3=importdata(route("1D/","Schwarz/","Sigma",NN,"10"));
Add_Data=importdata(route("1D/","Non_Schwarz/","AddData",NN,"1"));
N=Add_Data(1);
a=Imp_Data(1,1);
b=Imp_Data(1,2);
pa=Imp_Data(2,1);
pb=Imp_Data(2,2);
E=Imp_Data(3,1);
nyu=Imp_Data(3,2);
uk=Imp_Data(4,1);
rk=Imp_Data(4,2);
h=(b-a)/N;
L=b-a;
y_an=@(r)(((1-2*nyu)*(1+nyu)/E)*((pa*(a^2)-pb*b^2)/(b^2-a^2))*r+((1+nyu)/E)*((a^2+b^2)/r)*...
    ((pa-pb)/(b^2-a^2))).*uk;
SigmaR_an=@(r)(((pa*(a.^2)-pb.*b.^2)./(b.^2-a.^2))-(((a.^2*b.^2)./(r.^2)).*((pa-pb)./(b.^2-a.^2)))).*rk;
SigmaT_an=@(r)(((pa.*a.^2-pb.*b.^2)/(b.^2-a.^2))+(((a.^2*b.^2)./(r.^2))*((pa-pb)./(b.^2-a.^2)))).*rk;
r=linspace(a,b,10);
rr=linspace(a,b,N);
ErrorL2=@(Name)sqrt(sum((Name(:,1).'-SigmaR_an(a:h:b-h)).^(2)))/sqrt(sum((SigmaR_an(a:h:b-h)).^2));
ErrorL2(Sigma_S3)

%SR1_1=fplot(SigmaR_an,[a b],"r-");
set( SR1_1, 'LineWidth', 2 );
hold on
%SR1_2=plot(r,Sigma_NS(1:N/10:end,1),"bo");
set( SR1_2, 'LineWidth', 2 );
axis auto
l1=legend("Analytical","Numerical","Location","southeast");
%saveas(SR1_2,"SRW_Text/graphs/SigmaR.png");
figure;
%SR2_1=fplot(SigmaT_an,[a b],"r-");
set( SR2_1, 'LineWidth', 2 );
hold on
%SR2_2=scatter(r,Sigma_NS(1:N/10:end,2));
set( SR2_2, 'LineWidth', 2);
axis auto
l2=legend("Analytical","Numerical","Location","northeast");
%saveas(SR2_2,"SRW_Text/graphs/SigmaT.png")

figure;
%scatter(r,Sigma_NS(1:N/10:end,1));
hold on
%scatter(r,Sigma_S1(1:N/10:end,1));
hold on
%scatter(r,Sigma_S2(1:N/10:end,1));
hold on
%scatter(r,Sigma_S3(1:N/10:end,1));

