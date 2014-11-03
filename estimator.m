function [Pfgivenc1, occurances]  = estimator(features,stemmed,estimator_type, thresholdF)
[occurances]  = cal_samples(features,stemmed);
total_occurances = sum(occurances);


switch estimator_type 
    
    case 0 %Maximum likelihood estimator
        Pfgivenc1 = (occurances )/(total_occurances);
    case 1 %Laplace estimator
        Pfc1 = (occurances + ones(size(occurances)))./(total_occurances+size(features{1},2));
        Pfgivenc1 = Pfc1 ./ sum (Pfc1);
    case 2 %Good-TUring estimator
%J give you repeated indexes, so the occurrences of J(ind)
%gives you the occurences of B(ind).
        Pfgivenc1 = Good_Turing(occurances, thresholdF);        
    end