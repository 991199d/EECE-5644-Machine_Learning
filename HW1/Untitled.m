clear all; close all;clc;

%Initialize Parameters and Generate Data
N = 10000;
n = 2;
p0 = 0.65;
p1 = 0.35;

%Determine posteriors
Postlabel =rand(1, N) >= p0;

%Create appropriate number of data points from each distribution
Data0 = length(find(Postlabel == 0));
Data1 = length(find(Postlabel == 1));


%Parameters for two classes
mean = [3,0;0,3];
Cov = cat(3,[2,0;0,1],[1,0;0,2]);
mean01 = [3;0];
mean02 = [0;3];
mean03 = (p0*mean01) + (p1*mean02);

weight =[0.5;0.5];
ab = gmdistribution(mean, Cov,weight);

mean1 = [2;2];
Cov1 = [1,0;0,1];

%Generate data as prescribed in assignment description
x0 = random(ab, Data0);
x1 = mvnrnd(mean1, Cov1, Data1);

%Plot data showing two classes
plot(x0(:,1),x0(:,2),'o',x1(:,1),x1(:,2),'+');
title('Class 0 and Class 1 True Class Labels')
xlabel('x_1'),ylabel('x_2')
legend('Class 0','Class 1')

x=zeros(N,2);
x(Postlabel==0,:)=x0;
x(Postlabel==1,:)=x1;
figure(1);

%% Part A - ERM with True Knowledge
% Calculate discriminant score
discriminantScore=log(evalGaussian(x' ,mean1,Cov1)./(pdf(ab,x))');
sortDS=sort(discriminantScore);

%Generate vector of gammas for parametric sweep
logGamma=[min(discriminantScore)-eps sort(discriminantScore)+eps];

for ind=1:length(logGamma)
 decision=discriminantScore>logGamma(ind);
 Num_pos(ind)=sum(decision);
 pFP(ind)=sum(decision==1 & Postlabel==0)/Data0;
 pTP(ind)=sum(decision==1 & Postlabel==1)/Data1;
 pFN(ind)=sum(decision==0 & Postlabel==1)/Data1;
 pTN(ind)=sum(decision==0 & Postlabel==0)/Data0;
 %Two ways to make sure I did it right
 pFE(ind)=(sum(decision==0 & Postlabel==1) + sum(decision==1 & Postlabel==0))/S;
 pFE2(ind)=(pFP(ind)*Data0 + pFN(ind)*Data1)/S;
end

%Calculate Theoretical Minimum Error
logGamma_ideal=log(p0/p1);
decision_ideal=discriminantScore>logGamma_ideal;
pFP_ideal=sum(decision_ideal==1 & Postlabel==0)/Data0;
pTP_ideal=sum(decision_ideal==1 & Postlabel==1)/Data1;
pFE_ideal=(pFP_ideal*Data0+(1-pTP_ideal)*Data1)/(Data0+Data1);
%Estimate Minimum Error
%If multiple minimums are found choose the one closest to the theoretical
%minimum
[min_pFE, min_pFE_ind]=min(pFE);
if length(min_pFE_ind)>1
 [~,minDistTheory_ind]=min(abs(logGamma(min_pFE_ind)-logGamma_ideal));
 min_pFE_ind=min_pFE_ind(minDistTheory_ind);
end
%Find minimum gamma and corresponding false and true positive rates
minGAMMA=exp(logGamma(min_pFE_ind));
min_FP=pFP(min_pFE_ind);
min_TP=pTP(min_pFE_ind);

%Plot
figure;
plot(pFP,pTP, 'b-','DisplayName','ROC Curve');
hold all;
plot(min_FP,min_TP,'o','DisplayName','Estimated Min. Error');
plot(pFP_ideal,pTP_ideal,'+','DisplayName',...
 'Theoretical Min. Error');
xlabel('P(D=1|L=0) False Positive');
ylabel('P(D=1|L=1) True Positive');
title('Mininimum Expected Risk ROC Curve');
legend 'show';
grid on; box on;


fprintf('Theoretical: Gamma=%1.2f, Error=%1.2f%%\n',...
    exp(logGamma_ideal),100*pFE_ideal);
fprintf('Estimated: Gamma=%1.2f, Error=%1.2f%%\n',minGAMMA,100*min_pFE);
figure;
plot(logGamma,pFE,'DisplayName','Errors','LineWidth',2);
hold on;
plot(logGamma(min_pFE_ind),pFE(min_pFE_ind),...
 'ro','DisplayName','Minimum Error','LineWidth',2);
xlabel('Gamma');
ylabel('Proportion of Errors');
title('Probability of Error vs. Gamma')
grid on;
legend 'show';
