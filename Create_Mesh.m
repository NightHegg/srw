clc
global node edge
format shortG

node=importdata("files/2D/nodes.dat");
edge=importdata("files/2D/edges.dat");
hfun = (max(node(:,1))-min(node(:,1)))/4;

[vert,etri,tria,tnum] = refine2(node,edge,[],[],hfun);
   
[vnew,etri,tnew,tnum] = smooth2(vert,etri,tria,tnum);

PlotMesh(vnew, tnew);
writematrix(vnew,"files/2D/triMesh.dat",'Delimiter',' ');
writematrix(tnew,'files/2D/triElements.dat','Delimiter',' ');
    
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
    
     
    