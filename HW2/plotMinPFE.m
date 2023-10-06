function plotMinPFE(logGamma,pFE,min_pFE_ind,logGamma_ideal,pFE_ideal)
figure;
plot(logGamma,pFE,'DisplayName','Errors','LineWidth',2);
hold on;
plot(logGamma(min_pFE_ind),pFE(min_pFE_ind),...
    'ro','DisplayName','Minimum Error','LineWidth',2);
hold on;
plot(logGamma_ideal,pFE_ideal,...
    '+g','DisplayName','Calculated Min.Error','LineWidth',2);
xlabel('Gamma');
ylabel('Proportion of Errors');
title('Probability of Error vs. Gamma')
grid on;
legend 'show';
end