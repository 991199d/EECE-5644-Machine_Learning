function [outputNet,outputPFE, optM,stats]=kfoldMLP_NN(numPerc,k,x,labels,numLabels)
%Assumes data is evenly divisible by partition choice which it should be
N=length(x);
numValIters=10;
%Create output matrices from labels
y=zeros(numLabels,length(x));
for ind=1:numLabels
 y(ind,:)=(labels==ind-1);
end
%Setup cross validation on training data
partSize=N/k;
partInd=[1:partSize:N length(x)];
%Perform cross validation to select number of perceptrons
for M=1:numPerc

 for ind=1:k
index.val=partInd(ind):partInd(ind+1);
 index.train=setdiff(1:N,index.val);

 %Create object with M perceptrons in hidden layer
 net=patternnet(M);
% net.layers{1}.transferFcn = 'softplus';%didn't work

 %Train using training data
 net=train(net,x(:,index.train),y(:,index.train));
 %Validate with remaining data
 yVal=net(x(:,index.val));

 [~,labelVal]=max(yVal);
 labelVal=labelVal-1;

 pFE(ind)=sum(labelVal~=labels(index.val))/partSize;

 end

 %Determine average probability of error for a number of perceptrons
 avgPFE(M)=mean(pFE);
 stats.M=1:M;
 stats.mPFE=avgPFE;

end
%Determine optimal number of perceptrons
[~,optM]=min(avgPFE);
%Train one final time on all the data
for ind=1:numValIters
 netName(ind)={['net' num2str(ind)]};
 finalnet.(netName{ind})=patternnet(optM);
% finalnet.layers{1}.transferFcn = 'softplus';%Set to RELU

 finalnet.(netName{ind})=train(net,x,y);

 yVal=finalnet.(netName{ind})(x);
 [~,labelVal]=max(yVal);
 labelVal=labelVal-1;

 pFEFinal(ind)=sum(labelVal~=labels)/length(x);

end
[minPFE,outInd]=min(pFEFinal);
stats.finalPFE=pFEFinal;
outputPFE=minPFE;
outputNet=finalnet.(netName{outInd});
end


