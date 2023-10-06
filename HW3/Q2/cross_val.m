function best_GMM=cross_val(x)
B=10;M=6;%repetitionsperdataset;maxGMMconsidered
perf_array=zeros(B,M);%savespaceforperformanceevaluation
%Testeachdataset10times
for b=1:B
    set_size=500;
    train_index=randi([1,length(x)],[1,set_size]);
    train_set=x(:,train_index)+(1e-3)*randn(2,set_size);
    val_index=randi([1,length(x)],[1,set_size]);
    val_set=x(:,val_index)+(1e-3)*randn(2,set_size);
    for m=1:M
        GMModel=fitgmdist(train_set',M,'RegularizationValue',1e-10);
        alpha=GMModel.ComponentProportion;
        mu=(GMModel.mu)';
        sigma=GMModel.Sigma;
        %Calculatelog􀀀likelihoodperformancewithnewparameters
        perf_array(b,m)=sum(log(evalGMM(val_set,alpha,mu,sigma)));
    end
end
avg_perf=sum(perf_array)/B;
best_GMM=find(avg_perf==max(avg_perf),1);
end