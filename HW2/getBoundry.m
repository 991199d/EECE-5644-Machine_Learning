function s = getBoundry(h, v, theta)
z = zeros(length(h), length(v));
for i = 1:length(h)
for j = 1:length(v)
x_bound = [1 h(i) v(j) h(i)^2 h(i)*v(j) v(j)^2];
z(i, j) = x_bound.*theta;
end
end
s = z';
end
