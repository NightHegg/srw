clc
global a b h N L rSpecific 
format shortG
%format loose
sep="_";
NN="050";
stopCrit="SC_1e-06/";
route=@(Dim,Type,obj,size,AS)"results/"+Dim+Type+obj+sep+size+AS+".dat";
Imp_Data=importdata("files/mainData.dat");
y=importdata(route("1D/","Non_Schwarz/","y",NN,""));
Sigma_NS=importdata(route("1D/","Non_Schwarz/","Sigma",NN,""));
Sigma_S1=importdata(route("1D/","Schwarz/"+stopCrit,"Sigma",NN,"_2"));
Sigma_S2=importdata(route("1D/","Schwarz/"+stopCrit,"Sigma",NN,"_4"));
Sigma_S3=importdata(route("1D/","Schwarz/"+stopCrit,"Sigma",NN,"_10"));
Add_Data=importdata(route("1D/","Non_Schwarz/","AddData",NN,""));
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
rSpecific=linspace(a,b,10);
rReal=linspace(a,b,N);

[ErrorL2(SigmaT_an, Sigma_NS, 2) ...
ErrorL2(SigmaT_an, Sigma_S1, 2) ...
ErrorL2(SigmaT_an, Sigma_S2, 2) ...
ErrorL2(SigmaT_an, Sigma_S3, 2) ]

% plotAn(SigmaR_an);
% hold on
% plotNum(Sigma_S2,1);
% %saveas(SR1_2,"SRW_Text/graphs/SigmaR.png");
% figure;
% plotAn(SigmaT_an);
% hold on
% plotNum(Sigma_S2,2);
% %saveas(SR2_2,"SRW_Text/graphs/SigmaT.png")

%figure;
%scatter(rSpecific,Sigma_NS(1:N/10:end,1));
%hold on
%scatter(rSpecific,Sigma_S1(1:N/10:end,1));
% hold on
% scatter(rSpecific,Sigma_S2(1:N/10:end,1));
% hold on
% scatter(rSpecific,Sigma_S3(1:N/10:end,1));

function res = plotAn(Name)
global a b
res = fplot(Name,[a b],"r-");
set( res, 'LineWidth', 2 );
end

function res = plotNum(Name,Num)
global N rSpecific
res=scatter(rSpecific,Name(1:N/10:end,Num),"b");
set( res, 'LineWidth', 2 );
axis auto
legend("Analytical","Numerical","Location","southeast");
end

function res=ErrorL2(NameAn,NameNum,Num)
global a b h L
si=h/L;
res=sqrt(sum((NameNum(:,Num).'-NameAn(a:h:b-h)).^2*si))./sqrt(sum((NameAn(a:h:b-h)).^2*si));
end



