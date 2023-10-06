%% Setup and Sample Generation
clear all;
close all;
global C mu sigma p n %will need these in functions
nTrain1 = 100; %number of samples
nTrain2 = 1000;
nTrain3 = 10000;
nValidate = 20000;
n = 2; %number of dimensions
C = 2; %number of classes
% Class priors and class conditional distributions
p = [0.6, 0.4]; %class priors
mu(:,1) = [5; 0;]; %m01
mu(:,2) = [0; 4;]; %m02
mu(:,3) = [3; 2;]; %m1
sigma(:,:,1) = [4 0
 0 2]; %C01
sigma(:,:,2) = [1 0
 0 3]; %C02
sigma(:,:,3) = [2 0
 0 2]; %C1
% Generate and label different datasets
[label1, x1, NumClass1] = dataGeneration(nTrain1);
[label2, x2, NumClass2] = dataGeneration(nTrain2);
[label3, x3, NumClass3] = dataGeneration(nTrain3);
[label4, x4, NumClass4] = dataGeneration(nValidate);
%% Part 1 - ERM with True Knowledge on Validation Set
% Evaluate class conditional pdfs
pxgivenl(1,:) = .5*mvnpdf(x4', mu(:,1)', sigma(:,:,1))' + .5*mvnpdf(x4', mu(:,2)', sigma(:,:,2))'; %two distributions for class 0
pxgivenl(2,:) = mvnpdf(x4', mu(:,3)', sigma(:,:,3))';
% Calculate discriminant score
dscScr = log(pxgivenl(2,:)) - log(pxgivenl(1,:));
% Vary gamma from (effectively) 0 to inf, and take ln
logGamma = log(logspace(-18,18,1000));
% For each gamma, calculate TP, TN, FP and FN rates
for i = 1:length(logGamma)
 decision = dscScr > logGamma(i);
 pTP(i) = sum(decision==1 & label4==1)/NumClass4(2);
 pTN(i) = sum(decision==0 & label4==0)/NumClass4(1);
 pFP(i) = sum(decision==1 & label4==0)/NumClass4(1);
 pFN(i) = sum(decision==0 & label4==1)/NumClass4(2);
 pE(i) = pFP(i)*p(1) + pFN(i)*p(2); %error probability
end
% Plot ROC
figure
plot(pFP, pTP, 'b-', 'DisplayName', 'ROC Curve')
hold on
title("Expected Risk Minimization ROC Curve")
xlabel("P(D=1|L=0) False Positive")
ylabel("P(D=1|L=1) True Positive")
grid on
% Find minimum error threshold from above and add to plot
[minpE, minpEind] = min(pE);
plot(pFP(minpEind), pTP(minpEind), 'rx', 'DisplayName', 'Estimated Min. Error')
legend('Location', 'southeast')
hold off
% Plot true Validation dataset
figure
plot(x4(1, label4==0),x4(2, label4==0),'bo', 'DisplayName', 'Class 0')
hold on
plot(x4(1, label4==1),x4(2, label4==1),'rx', 'DisplayName', 'Class 1')
title('Validation Dataset True Distributions')
legend
hold off
% Plot data according to whether it was classified correctly, for optimal gamma
global shapes x1grid x2grid x1gridMatrix x2gridMatrix %will use these for all plotting
shapes = ['o';'x'];
% For decision boundaries
x1grid = linspace(floor(min(x4(1,:))-5), ceil(max(x4(1,:))+5));
x2grid = linspace(floor(min(x4(2,:))-5), ceil(max(x4(2,:))+5));
[x1gridMatrix,x2gridMatrix] = meshgrid(x1grid,x2grid);
decision = dscScr > logGamma(minpEind); % at optimal gamma
plotData(x4, decision, label4, logGamma(minpEind), 'ERM', 'ERM Correct vs. Incorrect Classification');
%% Part 2a - MLE of Training Datasets and Classification - Logistic Linear
% Calculate z(x) for each dataset
zLinearTrain1(:,:) = [ones(1, nTrain1);
 x1(1,:);
 x1(2,:);];
zLinearTrain2(:,:) = [ones(1, nTrain2);
 x2(1,:);
 x2(2,:);];
zLinearTrain3(:,:) = [ones(1, nTrain3);
 x3(1,:);
 x3(2,:);];
zLinearValidate(:,:) = [ones(1, nValidate);
 x4(1,:);
 x4(2,:);];
% Set initial guess for theta
thetaLinearInitial = zeros(3,1);
% Train on each of the datasets
options = optimset('MaxFunEvals',10000,'MaxIter',10000,'TolFun',1e-8,'TolX',1e-8);
thetaLinearTrain1 = fminsearch(@(t)MLEmodel(t, zLinearTrain1, label1, nTrain1),thetaLinearInitial,options);
thetaLinearTrain2 = fminsearch(@(t)MLEmodel(t, zLinearTrain2, label2, nTrain2),thetaLinearInitial,options);
thetaLinearTrain3 = fminsearch(@(t)MLEmodel(t, zLinearTrain3, label3, nTrain3),thetaLinearInitial,options);
% Use trained data to make decision on validation set
decisionLinear1 = thetaLinearTrain1.'*zLinearValidate > 0;
decisionLinear2 = thetaLinearTrain2.'*zLinearValidate > 0;
decisionLinear3 = thetaLinearTrain3.'*zLinearValidate > 0;
% Calculate probabilities of error
pELinear1 = (sum(decisionLinear1==0 & label4==1)+sum(decisionLinear1==1 & label4==0))/nValidate;
pELinear2 = (sum(decisionLinear2==0 & label4==1)+sum(decisionLinear2==1 & label4==0))/nValidate;
pELinear3 = (sum(decisionLinear3==0 & label4==1)+sum(decisionLinear3==1 & label4==0))/nValidate;
% Plot data according to whether it was classified correctly for each training set
plotData(x4, decisionLinear1, label4, thetaLinearTrain1, 'L', 'Logistic Linear Training Set D^1^0^0_T_r_a_i_n Correct vs. Incorrect Classification');plotData(x4, decisionLinear2, label4, thetaLinearTrain2, 'L', 'Logistic Linear Training Set D^1^0^0^0_T_r_a_i_n Correct vs. Incorrect Classification');
plotData(x4, decisionLinear3, label4, thetaLinearTrain3, 'L', 'Logistic Linear Training Set D^1^0^0^0^0_T_r_a_i_n Correct vs. Incorrect Classification');
%% Part 2b - MLE of Trainig Datasets and Classification - Logistic Quadratic
% Calculate z(x) for each dataset
zQuadraticTrain1(:,:) = [ones(1, nTrain1);
 x1(1,:);
 x1(2,:);
 x1(1,:).^2;
 x1(1,:).*x1(2,:);
 x1(2,:).^2;];
zQuadraticTrain2(:,:) = [ones(1, nTrain2);
 x2(1,:);
 x2(2,:);
 x2(1,:).^2;
 x2(1,:).*x2(2,:);
 x2(2,:).^2;];
zQuadraticTrain3(:,:) = [ones(1, nTrain3);
 x3(1,:);
 x3(2,:);
 x3(1,:).^2;
 x3(1,:).*x3(2,:);
 x3(2,:).^2;];
zQuadraticValidate(:,:) = [ones(1, nValidate);
 x4(1,:);
 x4(2,:);
 x4(1,:).^2;
 x4(1,:).*x4(2,:);
 x4(2,:).^2;];
% Set initial guess for theta
thetaQuadraticInitial = zeros(6,1);
% Train on each of the datasets
thetaQuadraticTrain1 = fminsearch(@(t)MLEmodel(t, zQuadraticTrain1, label1, nTrain1),thetaQuadraticInitial,options);
thetaQuadraticTrain2 = fminsearch(@(t)MLEmodel(t, zQuadraticTrain2, label2, nTrain2),thetaQuadraticInitial,options);
thetaQuadraticTrain3 = fminsearch(@(t)MLEmodel(t, zQuadraticTrain3, label3, nTrain3),thetaQuadraticInitial,options);
% Use trained data to make decision on validation set
decisionQuadratic1 = thetaQuadraticTrain1.'*zQuadraticValidate > 0;
decisionQuadratic2 = thetaQuadraticTrain2.'*zQuadraticValidate > 0;
decisionQuadratic3 = thetaQuadraticTrain3.'*zQuadraticValidate > 0;
% Calculate probabilities of error
pEQuadratic1 = (sum(decisionQuadratic1==0 & label4==1)+sum(decisionQuadratic1==1 & label4==0))/nValidate;
pEQuadratic2 = (sum(decisionQuadratic2==0 & label4==1)+sum(decisionQuadratic2==1 & label4==0))/nValidate;
pEQuadratic3 = (sum(decisionQuadratic3==0 & label4==1)+sum(decisionQuadratic3==1 & label4==0))/nValidate;
% Plot data according to whether it was classified correctly for each training set
plotData(x4, decisionQuadratic1, label4, thetaQuadraticTrain1, 'Q', 'Logistic Quadratic Training Set D^1^0^0_T_r_a_i_n Correct vs. Incorrect Classification');
plotData(x4, decisionQuadratic2, label4, thetaQuadraticTrain2, 'Q', 'Logistic Quadratic Training Set D^1^0^0^0_T_r_a_i_n Correct vs. Incorrect Classification');
plotData(x4, decisionQuadratic3, label4, thetaQuadraticTrain3, 'Q', 'Logistic Quadratic Training Set D^1^0^0^0^0_T_r_a_i_n Correct vs. Incorrect Classification');
%% Cleanup
clearvars -except minpE pELinear1 pELinear2 pELinear3 pEQuadratic1 pEQuadratic2 pEQuadratic3thetaLinearTrain1 thetaLinearTrain2 thetaLinearTrain3 thetaQuadraticTrain1 thetaQuadraticTrain2thetaQuadraticTrain3