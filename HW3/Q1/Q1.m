clear all;
close all;
%Switches to bypass parts 1 and 2 for debugging
dimensions=3; %Dimension of data
numLabels=4;
Lx={'L0', 'L1','L2','L3'};
% For min-Perror design, use 0-1 loss
lossMatrix = ones(numLabels,numLabels)-eye(numLabels);
muScale=2.45;
SigmaScale=0.2;
%Define data
D.d100.N=100;
D.d200.N=200;
D.d500.N=500;
D.d1k.N=1e3;
D.d2k.N=2e3;
D.d5k.N=5e3;
D.d100k.N=100e3;
dTypes=fieldnames(D);
%Define Statistics
p=ones(1,numLabels)/numLabels; %Prior
%Label data stats
mu.L0=muScale*[1 1 0]';
RandSig=SigmaScale*rand(dimensions,dimensions);
Sigma.L0(:,:,1)=RandSig*RandSig'+eye(dimensions);
mu.L1=muScale*[0 1 0]';
RandSig=SigmaScale*rand(dimensions,dimensions);
Sigma.L1(:,:,1)=RandSig*RandSig'+eye(dimensions);
mu.L2=muScale*[0 0 1]';
RandSig=SigmaScale*rand(dimensions,dimensions);
Sigma.L2(:,:,1)=RandSig*RandSig'+eye(dimensions);
mu.L3=muScale*[1 0 1]';
RandSig=SigmaScale*rand(dimensions,dimensions);
Sigma.L3(:,:,1)=RandSig*RandSig'+eye(dimensions);
%%Generate Data%%
for ind=1:length(dTypes)
 D.(dTypes{ind}).x=zeros(dimensions,D.(dTypes{ind}).N); %Initialize Data

 [D.(dTypes{ind}).x,D.(dTypes{ind}).labels,...
 D.(dTypes{ind}).N_l,D.(dTypes{ind}).p_hat]=...
 genData(D.(dTypes{ind}).N,p,mu,Sigma,Lx,dimensions);

end
%Plot Training Data
figure;
for ind=1:length(dTypes)-1
 subplot(3,2,ind);
 plotData(D.(dTypes{ind}).x,D.(dTypes{ind}).labels,Lx);
 legend 'show';
 title([dTypes{ind}]);
end
%Plot Validation Data
figure;
plotData(D.(dTypes{ind}).x,D.(dTypes{ind}).labels,Lx);
legend 'show';
title([dTypes{end}]);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Determine Theoretically Optimal Classifier
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for ind=1:length(dTypes)

 [D.(dTypes{ind}).opt.PFE, D.(dTypes{ind}).opt.decisions]=...
 optClass(lossMatrix,D.(dTypes{ind}).x,mu,Sigma,...
 p,D.(dTypes{ind}).labels,Lx);

 opPFE(ind)=D.(dTypes{ind}).opt.PFE;

 fprintf('Optimal pFE, N=%1.0f: Error=%1.2f%%\n',...
 D.(dTypes{ind}).N,100*D.(dTypes{ind}).opt.PFE);

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Train and Validate Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
numPerc=15; %Max number of perceptrons to attempt to train
k=10; %number of folds for kfold validation
for ind=1:length(dTypes)-1

 %kfold validation is in this function
 [D.(dTypes{ind}).net,D.(dTypes{ind}).minPFE,...
 D.(dTypes{ind}).optM,valData.(dTypes{ind}).stats]=...
 kfoldMLP_NN(numPerc,k,D.(dTypes{ind}).x,...
 D.(dTypes{ind}).labels,numLabels);

 %Produce validation data from test dataset
 valData.(dTypes{ind}).yVal=D.(dTypes{ind}).net(D.d100k.x);

 [~,valData.(dTypes{ind}).decisions]=max(valData.(dTypes{ind}).yVal);
 valData.(dTypes{ind}).decisions=valData.(dTypes{ind}).decisions-1;

 %Probability of Error is wrong decisions/num data points
 valData.(dTypes{ind}).pFE=...
 sum(valData.(dTypes{ind}).decisions~=D.d100k.labels)/D.d100k.N;

 outpFE(ind,1)=D.(dTypes{ind}).N;
 outpFE(ind,2)=valData.(dTypes{ind}).pFE;
 outpFE(ind,3)=D.(dTypes{ind}).optM;

 fprintf('NN pFE, N=%1.0f: Error=%1.2f%%\n',...
 D.(dTypes{ind}).N,100*valData.(dTypes{ind}).pFE);

end
%This code was used to plot the results from the data generated in the main
%function
%Extract cross validation results from structure
for ind=1:length(dTypes)-1
 [~,select]=min(valData.(dTypes{ind}).stats.mPFE);
 M(ind)=(valData.(dTypes{ind}).stats.M(select));
 N(ind)=D.(dTypes{ind}).N;

end
%Plot number of perceptrons vs. pFE for the cross validation runs
for ind=1:length(dTypes)-1
 figure;
 stem(valData.(dTypes{ind}).stats.M,valData.(dTypes{ind}).stats.mPFE);
 xlabel('Number of Perceptrons');
 ylabel('pFE');
 title(['Probability of Error vs. Number of Perceptrons for ' dTypes{ind}]);

end
%Number of perceptrons vs. size of training dataset
figure,semilogx(N(1:end-1),M(1:end-1),'o','LineWidth',2)
grid on;
xlabel('Number of Data Points')
ylabel('Optimal Number of Perceptrons')
ylim([0 10]);
xlim([50 10^4]);
title('Optimal Number of Perceptrons vs. Number of Data Points');
%Prob. of Error vs. size of training data set
figure,semilogx(outpFE(1:end-1,1),outpFE(1:end-1,2),'o','LineWidth',2)
xlim([90 10^4]);
hold all;semilogx(xlim,[opPFE(end) opPFE(end)],'r--','LineWidth',2)
legend('NN pFE','Optimal pFE')
grid on
xlabel('Number of Data Points')
ylabel('pFE')
title('Probability of Error vs. Data Points in Training Data');