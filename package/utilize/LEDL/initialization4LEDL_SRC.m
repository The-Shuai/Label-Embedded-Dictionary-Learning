function [Q] = initialization4LEDL_SRC(kernel_train,nclass,K)
% nclass 一共有多少类
% sample_numPerClass 每一类有多少训练样本
% dict_numPerClass 每一类多少字典

dict_numPerClass = round(K/nclass);
sample_numPerClass = round(size(kernel_train,2)/nclass);
dictLabel = [];
sampleLabel = [];
for class = 1:nclass
    labelvector = zeros(nclass,1);
    labelvector(class) = 1;
    dictLabel = [dictLabel repmat(labelvector,1,dict_numPerClass)];
    sampleLabel = [sampleLabel repmat(labelvector,1,sample_numPerClass)];
end

Q = zeros(K,size(kernel_train,2)); 
for frameid=1:size(kernel_train,2)
    label_training = sampleLabel(:,frameid);
    [maxv1,maxid1] = max(label_training);
    for itemid=1:K
        label_item = dictLabel(:,itemid);
        [maxv2,maxid2] = max(label_item);
        if(maxid1==maxid2)
            Q(itemid,frameid) = 1;
        else
            Q(itemid,frameid) = 0;
        end
    end
end

