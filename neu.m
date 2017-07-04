clear
load cora/graph.txt
load cora/group.txt
train_ratio=0.1; % train label ratio for classification task
%% hyperparameters
maxIter = 3;
lambda1=0.5;
lambda2=0.25;
C=5; %hyperparameter for svm; C=5 for cora dataset C=100 for blogcatalog and flickr

%% compute row-normalized adjacency matrix A
graph = graph+1; % let id start from 1 for Cora dataset
numOfNode = max(max(graph));
graph = [graph;graph(:,2) graph(:,1)];
A = sparse(graph(:,1),graph(:,2),1,numOfNode,numOfNode); %adjacency matrix
numOfGroup = 7;
group = group + 1;
group = sparse(group(:,1),group(:,2),1,numOfNode,numOfGroup);
grouptmp=group;
degree = sum(A,2);
A=sparse(graph(:,1),graph(:,2),1./degree(graph(:,1)),numOfNode,numOfNode); % row-normalized adjency matrix

%% load deepwalk embeddings
load cora/cora_deepwalk.txt
features = cora_deepwalk; 

%% test before NEU
% feature normalization; unnecessary for link prediction task
for i=1:size(features,2)
    if (norm(features(:,i))>0)
        features(:,i) = features(:,i)/norm(features(:,i));
    end
end

acc=0;
for i=1:10  % do the procedure for 10 times and take the average
    rp = randperm(numOfNode);
    testId = rp(1:floor(numOfNode*(1-train_ratio)));

    groupTest = group(testId,:);
    group(testId,:)=[];

    trainId = [1:numOfNode]';
    trainId(testId,:)=[];

    result=SocioDim(features, group, trainId, testId, C);
    [res,b] = evaluate(result,groupTest);
    acc=acc+res.micro_F1;
    group=grouptmp;
end
acc=acc/10

%% NEU
features = cora_deepwalk; % initial value
% feature normalization; unnecessary for link prediction task
for i=1:size(features,2)
    if (norm(features(:,i))>0)
        features(:,i) = features(:,i)/norm(features(:,i));
    end
end

tic;
for iter=1:maxIter
features=features+lambda1*A*features+lambda2*A*(A*features);
end
toc;

% feature normalization; unnecessary for link prediction task
for i=1:size(features,2)
    if (norm(features(:,i))>0)
        features(:,i) = features(:,i)/norm(features(:,i));
    end
end

%% test after NEU
acc=0;
for i=1:10  % do the procedure for 10 times and take the average
    rp = randperm(numOfNode);
    testId = rp(1:floor(numOfNode*(1-train_ratio)));

    groupTest = group(testId,:);
    group(testId,:)=[];

    trainId = [1:numOfNode]';
    trainId(testId,:)=[];

    result=SocioDim(features, group, trainId, testId, C);
    [res,b] = evaluate(result,groupTest);
    acc=acc+res.micro_F1;
    group=grouptmp;
end
acc=acc/10