function [data,labels] = generateMultiringDataset(numberOfClasses,numberOfSamples)

C = numberOfClasses;
N = numberOfSamples;
% Generates N samples from C ring-shaped 
% class-conditional pdfs with equal priors
ClassPriors = [0.35,0.65];
r=[2,4];
sigma = 1;
% Randomly determine class labels for each sample
thr = linspace(0,1,C+1); % split [0,1] into C equal length intervals
u = rand(1,N); % generate N samples uniformly random in [0,1]
labels = zeros(1,N);
for l = 1:C
    ind_l = find(thr(l)<u & u<=thr(l+1));
    labels(ind_l) = (rand(1,N) > ClassPriors(l-1));
    N_l = length(ind_l);
    theta_l = 2*pi*rand(1,N_l);
    x_l = sigma*randn(C,N_l) + r0*[cos(theta_l);sin(theta_l)];
    N.features(:,ind_l) = x_l; 
end
data = N.features;
if 1
    colors = rand(C,3);
    figure(1), clf,
    for l = 1:C
        ind_l = find(labels==l);
        plot(data(1,ind_l),data(2,ind_l),'.','MarkerFaceColor',colors(l,:)); axis equal, hold on,
    end
end
clear all, close all,