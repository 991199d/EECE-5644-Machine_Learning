function plotData(x,labels,Lx)
%Plots data
for ind=1:length(Lx)
 pindex=labels==ind-1;
 plot3(x(1,pindex),x(2,pindex),x(3,pindex),'.','DisplayName',Lx{ind});
 hold all;
end
grid on;
xlabel('x1');
ylabel('x2');
zlabel('x3');
end
