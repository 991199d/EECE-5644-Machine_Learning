function prob=CalcProb(discScore,logGamma,labels,N0,N1,phat)
for ind=1:length(logGamma)
	prob.decisions=discScore>=logGamma(ind);
	Num_pos(ind)=sum(prob.decisions);
	prob.p10(ind)=sum(prob.decisions==1 & labels==0)/N0;
	prob.p11(ind)=sum(prob.decisions==1 & labels==1)/N1;
	prob.p01(ind)=sum(prob.decisions==0 & labels==1)/N1;
	prob.p00(ind)=sum(prob.decisions==0 & labels==0)/N0;
 
 prob.pFE(ind)=prob.p10(ind)*phat(1) + prob.p01(ind)*phat(2);
end
end
