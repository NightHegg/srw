clc
clear all
yResult=importdata("results/2D/Non_Schwarz/y_019.dat");
[ySize,val]=size(yResult);
for i=1:ySize/2
yResult_mod(i,1)=yResult((i-1)*2+1);
yResult_mod(i,2)=yResult(i*2);
end
mesh=importdata("files/2D/mesh.dat");
modifiedMesh=mesh+yResult_mod;
elems=importdata("files/2D/elements.dat");
patch('Faces',elems,'Vertices',mesh,'FaceColor','none','EdgeColor','red','LineWidth',1.5,'LineStyle','--');
patch('Faces',elems,'Vertices',modifiedMesh,'FaceColor','none','EdgeColor','black','LineWidth',1.5);