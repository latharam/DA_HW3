function [precision, recall] = Precision_recall(ground_truth,predicted, class_labels)

%recall = zeros(size(class_labels,2));
 for i = 1:1:size(class_labels,2)
%      g = zeros(size(class_labels,1));
%      p = zeros(size(class_labels,1));
    g = (ground_truth == class_labels(i));
    p = (predicted == class_labels(i));
   
    false_positive = sum(~g.*p);
    false_negative = sum(g.*~p);
    true_positive = sum(g.*p);
    true_negetive = sum(~g.*p);
    
    precision(i) = (true_positive)/(true_positive+false_positive);
    recall(i) =  true_positive/(false_negative+true_positive);
    
 end
 
end
