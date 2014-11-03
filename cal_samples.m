function [occurances]  = cal_samples(features,stemmed)

%[B,I,J] = unique(stemmed,'rows');
[B,I,J] = unique(stemmed);
%J give you repeated indexes, so the occurrences of J(ind)
%gives you the occurences of B(ind).
counts_cd = histc(J, 1:length(B));
occurances = zeros(size(features));
for j = 1 : 1: size(features,2)
    for i = 1 : 1: size(B,2)
       if(strcmp(B{i},features{j}))
          occurances(j) = counts_cd(i);
          break;
       end
    end
end
end
