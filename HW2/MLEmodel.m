% MLE model derived in report
function out = MLEmodel(theta, x, label, N)
 x1gridMatrix = 1./(1+exp(-theta.'*x)); %logistic function
 out = (-1/N)*(sum((label*log(x1gridMatrix).')+(1-label)*log(1-x1gridMatrix).'));
end
