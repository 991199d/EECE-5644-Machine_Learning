function [label, x, NumClass] = dataGeneration(N)
 global mu sigma p n
 label = rand(1,N);
 for i = 1:length(label)
 if label(i) < p(1)/2 %two subclasses for the Class 0, will be combined later
 label(i) = 0;
 elseif label(i) < p(1)
 label(i) = 0.5;
 else
 label(i) = 1;
 end
 end
 NumClass = [sum(label==0),sum(label==0.5),sum(label==1)];
 x = zeros(n,N);
 x(:, label==0) = mvnrnd(mu(:,1), sigma(:,:,1), NumClass(1))';
 x(:, label==0.5) = mvnrnd(mu(:,2), sigma(:,:,2), NumClass(2))';
 x(:, label==1) = mvnrnd(mu(:,3), sigma(:,:,3), NumClass(3))';
 % Combine labels 0 and 0.5 into one class under label 0
 for i = 1:length(label)
 if label(i) == 0.5
 label(i) = 0;
 end
 end
 NumClass = [sum(label==0),sum(label==1)];
end

