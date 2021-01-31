clc
global node edge
format shortG
%----------------
dimTask="2D";
N=10;
amntElem=2;
%----------------
if N<10
    strN="00"+int2str(N);
elseif N<100
    strN="0"+int2str(N);
else
    strN=int2str(N);
end
%----------------
if dimTask=="2D" 
    node=importdata("files/2D/nodes.dat");
    edge=importdata("files/2D/edges.dat");
    hfun = (max(node(:,1))-min(node(:,1)))/amntElem;

    [vert,etri,tria,tnum] = refine2(node,edge,[],[],hfun);
   
    [meshAr,etri,elemAr,tnum] = smooth2(vert,etri,tria,tnum);
    PlotMesh(meshAr, elemAr);
    
    fileID1=fopen("files/2D/mesh.dat","w");
    fileID2=fopen("files/2D/elements.dat","w");
    
    fprintf(fileID1,'%g %g\n',meshAr(1:end-1,:)');
    fprintf(fileID1,'%g %g',meshAr(end,:)');
    
    fprintf(fileID2,'%g %g %g\n',elemAr(1:end-1,:)');
    fprintf(fileID2,'%g %g %g',elemAr(end,:)');
elseif dimTask=="1D"
    node=importdata("files/1D/nodes.dat");
    meshAr=linspace(node(1),node(2),N+1);
    fileID=fopen("files/1D/mesh_"+strN+".dat","w");
    fprintf(fileID,'%f\n',meshAr(:,1:end-1));
    fprintf(fileID,'%f',meshAr(:,end));
end
    
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
    
     
    