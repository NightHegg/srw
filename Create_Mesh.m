clc
global node edge

Imp_Data=importdata("files/mainData.dat");

node = [0,0;2,0;2,2;0,2];
edge=[1 2; 2 3; 3 4; 4 1];

hfun = .5 ;
[vert,etri,tria,tnum] = refine2(node,edge,[],[],hfun);
   
[vnew,etri,tnew,tnum] = smooth2(vert,etri,tria,tnum);
PlotMesh(vnew, tnew);
writematrix(vnew,"files/mesh.dat",'Delimiter',' ');
writematrix(tnew,'files/elements.dat','Delimiter',' ');
    
function res = PlotMesh(vert,tria)
    global node  edge
    figure;
    res=patch('faces',tria(:,1:3),'vertices',vert, ...
        'facecolor','w', ...
        'edgecolor',[.2,.2,.2]) ;
hold on; axis image off;
patch('faces',edge(:,1:2),'vertices',node, ...
         'facecolor','w', ...
         'edgecolor',[.1,.1,.1], ...
         'linewidth',1.5) ;
    end
    
     
    