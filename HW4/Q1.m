%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%EECE5644 Summer 2 2020
%Homework #4
%Problem #2
%Significant parts of this code were derived from the following sources
%g/code/svmKfoldCrossValidation.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%K-fold Validation of a SVM Classifier Using Gaussian Kernel
%Switches for new data and for skipping training
if 1 %Set to 0 to plot existing data
 if 0 %Set to 0 to repeat hyperparameter search on existing data
 clear all;
 close all;

 numL=2;
 k=10;
 
 D.train.N=1e3;
 D.test.N=10e3;
 
 [D.train.x,D.train.labels] = generateMultiringDataset(numL,D.train.N);
 [D.test.x,D.test.labels] = generateMultiringDataset(numL,D.test.N); 
 D.train.labels(D.train.labels==0)=-1;
 D.train.labels(D.train.labels==1)=1;
 
 D.test.labels(D.test.labels==0)=-1;
 D.test.labels(D.test.labels==1)=1;
 end
%Cross Validation to select parameters
%Setup cross validation on training data
partSize=D.train.N/k;
partInd=[1:partSize:D.train.N length(D.train.x)+1];
%Hyperparameter search parameters
sigmaList=logspace(-1,-0.8,40);
cList=logspace(1.4,1.6,40);
%Perform cross validation to select model parameters
avgPFE=zeros(length(cList),length(sigmaList));
for SigInd=1:length(sigmaList)
 for Cind=1:length(cList)
 
 for ind=1:k 
 index.val=partInd(ind):partInd(ind+1)-1;
 index.train=setdiff(1:D.train.N,index.val);
 
 %Train SVM using specified hyperparameters
 SVMk = fitcsvm(D.train.x(index.train)',D.train.labels(index.train),...
    'BoxConstraint',cList(Cind),'KernelFunction',...
 'gaussian','KernelScale',sigmaList(SigInd));
 
 %Validation decisions
 decisions = SVMk.predict(D.train.x(index.val)')';
% indCORRECT = lValidate.*dValidate == 1;
 indINCORRECT= D.train.labels(index.val).*decisions == -1;
% Ncorrect(k)=length(indCORRECT);
 pFE=sum(indINCORRECT)/length(index.val);
 
 end
 
 %Determine average probability of error for a number of perceptrons
 avgPFE(Cind,SigInd)=mean(pFE);
 
 fprintf('Sigma %1.0f/%1.0f, C %1.0f/%1.0f\n',SigInd,length(sigmaList),Cind,length(cList));
 
 end
end
end
%Plot results
figure; subplot(1,2,1);
contour(log10(cList),log10(sigmaList),1-avgPFE',20);
xlabel('log_{10} C');
ylabel('log_{10} sigma');
title('Gaussian-SVM Cross-Val Accuracy Estimate');
axis equal;
%Determine hyperparameter values that minimize prob. of error
[~,indMINpFE] = min(avgPFE(:)); 
[indOptC, indOptSigma] = ind2sub(size(avgPFE),indMINpFE);
cOpt= cList(indOptC); 
sigmaOpt= sigmaList(indOptSigma); 
%Train final model using entire training dataset
SVMopt = fitcsvm(D.train.x',D.train.labels','BoxConstraint',cOpt,...
 'KernelFunction','gaussian','KernelScale',sigmaOpt);
%Evaluate performance on test dataset
decisionsOpt=SVMopt.predict(D.test.x')';
decisionsEval=decisionsOpt.*D.test.labels;
dInc=decisionsEval==-1;
dCorr=decisionsEval==1;
pFEopt=sum(dInc)/D.test.N;
fprintf('Probability of Error = %1.2f%%\n',100*pFEopt);
%Plot correct and incorrect decisions
subplot(1,2,2);
plot(D.test.x(1,dCorr),D.test.x(2,dCorr),'go','DisplayName','Correct Decisions');
hold all;
plot(D.test.x(1,dInc),D.test.x(2,dInc),'r.','DisplayName','Incorrect Decisions');
grid on;
xlabel('x1'); ylabel('x2');
title(sprintf('Classification Decisions\nProbability of Error= %1.2f%%',100*pFEopt));
Nx = 1001; Ny = 990; 
xGrid = linspace(-10,10,Nx); 
yGrid = linspace(-10,10,Ny);
[h,v] = meshgrid(xGrid,yGrid); 
dGrid = SVMopt.predict([h(:),v(:)]); 
zGrid = reshape(dGrid,Ny,Nx);
figure(1), subplot(1,2,2);
contour(xGrid,yGrid,zGrid,0); 
xlabel('x1'), ylabel('x2'), axis equal;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%End of problem 2 code