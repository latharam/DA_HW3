%equal priors
tic
path = 'C:\Users\Administrator\Documents\MATLAB\DA_HW3\20_newsgroups';
no_samples = 30;
no_of_files = 100;
directories = dir(path);
max_kfold = 5;
no_of_features = 100;
features = ' ';
stemmed = ' ';
stemmed_test = ' ';
train_data = cell(max_kfold,1);
test_data  = cell(max_kfold,1);


stpwrd= fopen('english.stop');
Stop_words = textscan(stpwrd,'%s');
start_test = zeros(length(directories),no_of_files,max_kfold);
start_train = zeros(length(directories),no_of_files,max_kfold);
end_test = zeros(length(directories),no_of_files,max_kfold);
end_train = zeros(length(directories),no_of_files,max_kfold);

for i=3:length(directories)  %for each class
        foldername = char(directories(i).name);
        subdir_path = fullfile(path, foldername);
        myfiles = dir(subdir_path);
        myfiles = myfiles(3:size(myfiles));
        for kfold = 5:1:max_kfold   % changing the number of folds
           % c1 = cvpartition(size(myfiles,1),'kfold');
           c1 = cvpartition(no_of_files,'kfold');
           test_size(kfold) = size(myfiles(test(c1,1)),1);
            train_size(kfold) = size(myfiles(training(c1,1)),1);
            %test_size(kfold) = min(no_of_files,size(test(c1,1),1));
            %train_size(kfold) = no_of_files;
            stemmed = ' ';
            stemmed_test = ' ';
            for fold = 1:1:kfold  %predict for each fold 
                    train_set = myfiles(training(c1,fold));
                    test_set = myfiles(test(c1,fold));   
                    for j=1:1:train_size(kfold)   %for each sample
                    %for j=1:1:no_of_files
                       
                            filename = char(train_set(j).name);
                            filepath = fullfile(subdir_path, filename);
                            fp = fopen(filepath);
                            samples = textscan(fp,'%s',no_samples);
                            stemmed = [stemmed text_preprocessing(samples,Stop_words,1)];
                            start_train(i,j,fold) = size(train_data{fold},2);
                            train_data{fold} = [train_data{fold} , stemmed];
                            end_train(i,j,fold) = size(train_data{fold},2);
                            fclose(fp);
                            labels_train(fold,(i-3)*(train_size(kfold))+1:((i-2)*(train_size(kfold)))) = i.*ones(1,(train_size(kfold)));

                            %Pfgivenc1 (i) = Good_Turing(features(1), 

                            
                    %end
                    end
                            
                     
                     for j=1:1:test_size(kfold)   %for each sample
                    %for j=1:1:no_of_files
                       
                            filename = char(test_set(j).name);
                            filepath = fullfile(subdir_path, filename);
                            fp = fopen(filepath);
                            samples = textscan(fp,'%s',no_samples);
                            stemmed_test = [stemmed_test text_preprocessing(samples,Stop_words,1)];
                            start_test(i,j,fold) = size(test_data{fold},2);
                            test_data{fold} = [test_data{fold} , stemmed_test];
                            end_test(i,j,fold) = size(test_data{fold},2);
                            fclose(fp);
                            labels_test(fold,(i-3)*(test_size(kfold))+1:((i-2)*(test_size(kfold)))) = i.*ones(1,(test_size(kfold)));


                    end
                    
            end 
               if(i~=22)
                clear stemmed_test;
                clear stemmed;
               end
disp('done for directory\n')
disp(i)
           end
    
   end
%end

%PC = 0.5.*(ones(k,1));
[B,I,J] = unique([stemmed stemmed_test],'rows');
 %J give you repeated indexes, so the occurrences of J(ind)
 %gives you the occurences of B(ind).
counts = histc(J, 1:length(B));
mean_freq_cd= mean(counts);
%features_cd = B(1,counts_cd>((3/4)*mean_freq_cd));
[sorted_f, ind ]= sort(counts,'descend');
features = [features B(1,ind(1: no_of_features))];
features=unique(features);
clear sorted_f;
clear ind;
clear I;
clear J;
clear B;
clear stemmed_test;
clear stemmed;

 labels = zeros(max_kfold,no_of_files*(length(directories)-2));
 for fold = 1:1:kfold
  data_test = zeros(test_size(kfold)*(length(directories)-2),size(features,2));
  data_train = zeros(test_size(kfold)*(length(directories)-2),size(features,2));
 for i=3:length(directories)
     for l=1:1:train_size(kfold)   %for each file/sample
        data_train((i-3)*train_size(kfold)+l, : )  = cal_samples(features,train_data{fold}(start_train(i,l)+1:end_train(i,l)));
     end
    for l=1:1:test_size(kfold)   %for each file/sample
        data_test((i-3)*test_size(kfold)+l, : )  = cal_samples(features,test_data{fold}(start_test(i,l)+1:end_test(i,l)));
    end
    
% knn_labels(i-3:i-5) = [knn_labels ; i.*ones(no_of_files,1)];
 %labels(fold,(i-3)*(train_size(kfold)+test_size(kfold))+1:((i-2)*(train_size(kfold)+test_size(kfold)))) = i.*ones(1,(train_size(kfold)+test_size(kfold)));
 end
%[M  class(i,l)] = max(PC1givenD); for each sample max accross all
 %classes/dir

 data_knn = reshape(data_train,(train_size(kfold))*(length(directories)-2),size(features,2));
%  for p= 2:2:20
%  knn(p/2) = fitcknn(data_knn,labels_train(fold, :)','NumNeighbors',no_of_files/p);
%  end
 tree = fitctree(data_knn,labels_train(fold, :)');
 %nbc = fitcnb(data_knn,labels');
 nbc = fitNaiveBayes(data_knn,labels_train(fold, :)','Distribution','mn');
 
 for i = 1:1:(length(directories)-2)
     
     for l=1:1:(test_size(kfold))
     Xnew = data_test((i-1)*(test_size(kfold))+l, : );
     
     tree_labels((i-1)*(test_size(kfold))+l,fold)= predict(tree,Xnew);
     nbc_labels((i-1)*(test_size(kfold))+l,fold)= predict(nbc,Xnew);
     
     end
     
 end
 
for p= 1:1:10
     knn = fitcknn(data_knn,labels_train(fold, :)','NumNeighbors',2*p);
  for i = 1:1:(length(directories)-2)
     
     for l=1:1:(test_size(kfold))
      Xnew = data_test((i-1)*(test_size(kfold))+l, : );
      knn_labels((i-1)*(test_size(kfold))+l,fold,p)= predict(knn,Xnew);
     end
  end
end

 
 class_labels = 3:1:22;
 ground_truth(fold, : ) = labels_test(fold,:) ;
 for p= 1:1:10
     Prediction_knn(fold,:,p) = knn_labels(:,fold,p)' ;
    [precision_knn(fold,:,p), recall_knn(fold,:,p)] = Precision_recall(ground_truth(fold,:),Prediction_knn(fold,:,p), class_labels);
 end
 Prediction_tree(fold,:) = tree_labels(:,fold)' ;
[precision_tree(fold,:), recall_tree(fold,:)] = Precision_recall(ground_truth(fold,:),Prediction_tree(fold,:), class_labels);

 Prediction_nbc(fold,:) = nbc_labels(:,fold)' ;
[precision_nbc(fold,:), recall_nbc(fold,:)] = Precision_recall(ground_truth(fold,:),Prediction_nbc(fold,:), class_labels);

 
 end
 clear train_data;
 clear test_data;
 
%  ground_truth(fold) = labels_test(fold,:) ;
%  Prediction = knn_labels(:,fold)' ;
%  class_labels = 3:1:22;
% [precision_knn(fold, recall_knn(fold)] = Precision_recall(ground_truth(fold),Prediction(fold), class_labels);
% avg_precision_knn = sum(precision_knn)/size(precision_knn,2);
% avg_recall_knn = sum(recall_knn)/size(recall_knn,2);
% 
% ground_truth_tree = [labels_test(1,:) labels_test(2,:)];
%  Prediction_tree = [tree_labels(:,1)' tree_labels(:,2)'];
% [precision_tree, recall_tree] = Precision_recall(ground_truth_tree,Prediction_tree, class_labels);
% avg_precision_tree = sum(precision_tree)/size(precision_tree,2);
% avg_recall_tree = sum(recall_tree)/size(recall_tree,2);
% 
% ground_truth_nbc = [labels_test(1,:) labels_test(2,:)];
%  Prediction_nbc = [nbc_labels(:,1)' nbc_labels(:,2)'];
%  class_labels = 3:1:22;
% [precision_nbc, recall_nbc] = Precision_recall(ground_truth_nbc,Prediction_nbc, class_labels);
% avg_precision_nbc = sum(precision_nbc)/size(precision_nbc,2);
% avg_recall_nbc = sum(recall_nbc)/size(recall_nbc,2);
% 
f1_nbc= mean((2.*(precision_nbc.*recall_nbc)./(precision_nbc+recall_nbc+1)),1);
f1_tree= mean((2.*(precision_tree.*recall_tree)./(precision_tree+recall_tree+1)),1);
f1_knn1= mean((2.*(precision_knn(:).*recall_knn(:))./(precision_knn(:)+recall_knn(:)+1)),1);
f1_knn2= mean((2.*(precision_knn(:,:,1).*recall_knn(:,:,1))./(precision_knn(:,:,1)+recall_knn(:,:,1)+1)),1);

figure
plot(f1_nbc)
title('f1 score of nbc vs class');
xlabel('class');
ylabel('f1 score');
%figure
% plot(recall_nbc)
% title('recall of nbc vs fold ');
% xlabel('fold');
% ylabel('recall');
fclose(stpwrd);

% figure
% plot(recall_knn(:,:,1))
% title('recall of knn vs fold ');
% xlabel('fold');
% ylabel('recall');



figure
plot(f1_tree)
title('f1 score of tree vs class');
xlabel('class');
ylabel('f1 score');
%figure
% plot(recall_tree)
% title('recall of tree vs fold ');
% xlabel('fold');
% ylabel('recall');

figure
plot(f1_knn1)
title('f1 score of knn vs #neihbours ');
xlabel('fold');
ylabel('f1 score');
% figure
% plot(recall_knn(:))
% title('recall of knn vs #features ');
% xlabel('fold');
% ylabel('recall');


figure
plot(f1_knn2)
title('f1 score of knn vs fold ');
xlabel('fold');
ylabel('f1 score');

figure
plot(precision_nbc)
title('precion of nbc vs fold ');
xlabel('fold');
ylabel('precision');
figure
plot(recall_nbc)
title('recall of nbc vs fold ');
xlabel('fold');
ylabel('recall');


toc