function error = crossValidation(xTrain,yTrain, numComponents)
 cp = cvpartition(yTrain , 'KFold', 10);
 gm = @(xTrain, yTrain, xTest);
 predictGM(xTrain, yTrain, xTest, numComponents);
 cvMCR = crossval('mcr', xTrain, yTrain, 'predfun', gm, 'partition', cp); %misclassification error
 error = cvMCR;
end