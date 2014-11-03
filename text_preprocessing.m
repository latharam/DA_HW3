function stemmed = text_preprocessing(samples_cd,Stop_words,q)
%This function performs the text pre-processing

cd = char(samples_cd{1});

m=1;
 for k = 1:1:size(samples_cd{1,1},1)/q
     tokenizer = strsplit(samples_cd{1}{k},{'.','!','?', '"', '(', ')', '#', '[',']', '''','-', '*', '@', '>','<','%','$','&'});
       for l=1:1:size(tokenizer,2)
        if(~strcmp(tokenizer{l},''))
            lower_case = lower(tokenizer{l});
            stem_wrd =  porterStemmer(deblank(lower_case));
            flag =0;
            for i = 1: 1 : size(Stop_words{1,1},1)
               %if(strcmp(stemmed{k},Stop_words{1}{i}))
               if(strcmp(stem_wrd,Stop_words{1}{i}))
                   flag = 1;
                   break;
               end;        
            end
    
            if (flag == 0)
                stemmed{m}= stem_wrd; %deleting a cell array
                m =m+1;
            end;
        end
    end
 end



end