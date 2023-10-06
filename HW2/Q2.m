 clear all, close all,

% True position inside unit circle
radius = rand; theta = 2*pi*rand;
pTrue = [radius*cos(theta);radius*sin(theta)];   

for K = 1:4 % for each specified number of landmarks

    % Landmark positions evenly spaces on the unit circle
    radius = 1; theta = [0,2*pi/K*[1:(K-1)]];
    pLandmarks = [radius*cos(theta);radius*sin(theta)];

    % Generate range measurements
    sigma = 3e-1*ones(1,K);
    r = sqrt(sum((repmat(pTrue,1,K)-pLandmarks).^2,1)) + sigma.*randn(1,K);

    % Parameters of the prior
    sigmax = 25e-2; sigmay = sigmax;
    
    % Evaluate the MAP estimation objective function on a grid
    Nx = 101; Ny = 99;
    xGrid = linspace(-2,2,Nx); yGrid = linspace(-2,2,Ny);
    [h,v] = meshgrid(xGrid,yGrid);
    MAPobjective = (h(:)/sigmax).^2 + (v(:)/sigmay).^2;
    for i = 1:K
        di = sqrt((h(:)-pLandmarks(1,i)).^2+(v(:)-pLandmarks(2,i)).^2);
        MAPobjective = MAPobjective + ((r(i)-di)/sigma(i)).^2;
    end
    zGrid = reshape(MAPobjective,Ny,Nx);
    
    % Display true position and landmark positions
    figure(ceil(K/4)), subplot(2,2,mod(K-1,4)+1),
    plot(pTrue(1),pTrue(2),'+'); hold on,
    plot(pLandmarks(1,:),pLandmarks(2,:),'o'); axis([-2 2 -2 2]),
    % Display the MAP objective contours
    minV = min(MAPobjective); maxV = max(MAPobjective);
    values = minV + (sqrt(maxV-minV)*linspace(0.1,0.9,21)).^2;
    contour(xGrid,yGrid,zGrid,values); xlabel('x'), ylabel('y'),
    title(strcat({'MAP Objective for K = '},num2str(K))); 
    grid on, axis equal,
end