function out = testQ1(a, b, c, d, n)
    try
        mdl = fitgmdist(a, n);
    catch exception
        mdl = fitgmdist(a, n, 'SharedCov', true);
        disp('shared');
    end
    [~, out] = posterior(mdl, c);
end

