addpath /home/changmo/msr/DNN/DBN

%% 
% Step0: Set the parameters of the experiment
nSpeakers = 20;
nDims = 13;             % dimensionality of feature vectors
nMixtures = 32;         % How many mixtures used to generate data
nChannels = 250;         % Number of channels (sessions) per speaker
nFrames = 1000;         % Frames per speaker (10 seconds assuming 100 Hz)
nWorkers = 1;           % Number of parfor workers, if available

% Pick random centers for all the mixtures.
mixtureVariance = .10;
channelVariance = .05;
mixtureCenters = randn(nDims, nMixtures, nSpeakers);
channelCenters = randn(nDims, nMixtures, nSpeakers, nChannels)*.1;
trainSpeakerData = cell(nSpeakers, nChannels);
testSpeakerData = cell(nSpeakers, nChannels);
speakerID = zeros(nSpeakers, nChannels);

% Create the random data. Both training and testing data have the same
% layout.
for s=1:nSpeakers
    trainSpeechData = zeros(nDims, nMixtures);
    testSpeechData = zeros(nDims, nMixtures);
    for c=1:nChannels
        for m=1:nMixtures
            % Create data from mixture m for speaker s
            frameIndices = m:nMixtures:nFrames;
            nMixFrames = length(frameIndices);
            trainSpeechData(:,frameIndices) = ...
                randn(nDims, nMixFrames)*sqrt(mixtureVariance) + ...
                repmat(mixtureCenters(:,m,s),1,nMixFrames) + ...
                repmat(channelCenters(:,m,s,c),1,nMixFrames);
            testSpeechData(:,frameIndices) = ...
                randn(nDims, nMixFrames)*sqrt(mixtureVariance) + ...
                repmat(mixtureCenters(:,m,s),1,nMixFrames) + ...
                repmat(channelCenters(:,m,s,c),1,nMixFrames);
        end
        trainSpeakerData{s, c} = trainSpeechData;
        testSpeakerData{s, c} = testSpeechData;
        speakerID(s,c) = s;                 % Keep track of who this is
    end
end

%%
% Step1: Create the universal background model from all the training speaker data
nmix = nMixtures;           % In this case, we know the # of mixtures needed
final_niter = 10;
ds_factor = 1;
ubm = gmm_em(trainSpeakerData(:), nmix, final_niter, ds_factor, nWorkers);


%%
% Step2.1: Calculate the statistics needed for the iVector model.
stats = cell(nSpeakers, nChannels);
for s=1:nSpeakers
    for c=1:nChannels
        [N,F] = compute_bw_stats(trainSpeakerData{s,c}, ubm);
        stats{s,c} = [N; F];
    end
end

% Step2.2: Learn the total variability subspace from all the speaker data.
tvDim = 100;
niter = 5;
T = train_tv_space(stats(:), ubm, tvDim, niter, nWorkers);
%
% Now compute the ivectors for each speaker and channel.  The result is size
%   tvDim x nSpeakers x nChannels
devIVs = zeros(tvDim, nSpeakers, nChannels);
for s=1:nSpeakers
    for c=1:nChannels
        devIVs(:, s, c) = extract_ivector(stats{s, c}, ubm, T);
    end
end

%%
% step3.1
ldaDim = min(100, nSpeakers-1);
devIVbySpeaker = reshape(devIVs, tvDim, nSpeakers*nChannels);
[V,D] = lda(devIVbySpeaker, speakerID(:));
finalDevIVs = V(:, 1:ldaDim)' * devIVbySpeaker;

%%
% step3.2
nphi = ldaDim;
niter = 10;
pLDA = gplda_em(finalDevIVs, speakerID(:), nphi, niter);

%%
% step4.1
averageIVs = mean(devIVs, 3);
modelIVs = V(:, 1:ldaDim)' * averageIVs;

% Step4.2: Now compute the ivectors for the test set 
% and score the utterances against the models

testIVs = zeros(tvDim, nSpeakers, nChannels); 
for s=1:nSpeakers
    for c=1:nChannels
        [N, F] = compute_bw_stats(testSpeakerData{s, c}, ubm);
        testIVs(:, s, c) = extract_ivector([N; F], ubm, T);
    end
end
testIVbySpeaker = reshape(permute(testIVs, [1 3 2]), tvDim, nSpeakers*nChannels);
finalTestIVs = V(:, 1:ldaDim)' * testIVbySpeaker;



%%

order1 = 1:(nSpeakers*nChannels*3/5);
shuffle1 = order1(randperm(length(order1)));

order2 = 1:(nSpeakers*nChannels/5);
shuffle2 = order2(randperm(length(order2)));

order3 = 1:(nSpeakers*nChannels/5);
shuffle3 = order3(randperm(length(order3)));
%ivector_train = permute(devIVs,[2,3,1]);
%trainIV = reshape(ivector_train,[(nSpeakers*nChannels),100]);

%ivector_test = permute(testIVs, [2,3,1]);
%testIV = reshape(ivector_test,[(nSpeakers*nChannels),100]);

%ivector = permute(devIVs,[2,3,1]);
%IV = reshape(ivector,[(nSpeakers*nChannels),100]);

trainData = finalDevIVs(:,1:(nSpeakers*nChannels*3/5))';
testData = finalDevIVs(:,(nSpeakers*nChannels*3/5+1):(nSpeakers*nChannels*4/5))';
validData = finalDevIVs(:,(nSpeakers*nChannels*4/5+1):(nSpeakers*nChannels))';

%testIV = zeros((nSpeakers*nChannels/3),100);
classIV_train = zeros(nSpeakers*nChannels*3/5,nSpeakers);
classIV_test = zeros(nSpeakers*nChannels/5,nSpeakers);
classIV_valid = zeros(nSpeakers*nChannels/5,nSpeakers);

ord = 0;
ord1 = 0;
ord2 = 0;
ord3 = 0;

for q = 1:(nSpeakers*nChannels)
   
    if q <= (nSpeakers*nChannels*3/5)
        ord1 = ord1 + 1;
        classIV_train(ord1,mod(q-1,10)+1) = 1;
        classTrain(ord1,:)= mod(q-1,10);
    elseif (q >= (nSpeakers*nChannels*3/5+1)) && (q <= (nSpeakers*nChannels*4/5))
        ord2 = ord2 + 1;
        classIV_test(ord2,mod(q-1,10)+1) = 1;
        classTest(ord2,:) = mod(q-1,10);
    else
        ord3 = ord3 + 1;
        classIV_valid(ord3,mod(q-1,10)+1) = 1;
        classValid(ord3,:) = mod(q-1,10);
    end
    
    
end


csvwrite('skt_train_Data.csv',trainData)
csvwrite('skt_test_Data.csv',testData)
csvwrite('skt_valid_Data.csv',validData)
csvwrite('skt_train_taget.csv',classTrain)
csvwrite('skt_test_target.csv',classTest)
csvwrite('skt_valid_target.csv',classValid)
%classIV_train = zeros((nSpeakers*nChannels*2/3),nSpeakers);
%classIV_test = zeros((nSpeakers*nChannels/3),nSpeakers);
%classIV_train = zeros((nSpeakers*nChannels*2/3),nSpeakers);
%classIV_test = zeros((nSpeakers*nChannels/3),nSpeakers);

%for i = 1:nSpeakers
%    for j = 1:nChannels
        
%        ord = ord + 1;        
%        if j <= (nChannels*3/5)
%            ord1 = ord1 + 1;
            %classIV_train(shuffle1(ord1),i) = 1; 
%            classIV_train(shuffle1(ord1),:) = i-1;
%            trainData(shuffle1(ord1),:) = IV(ord,:); % 2000 x 100
%        elseif (j > (nChannels*3/5)) && (j <= (nChannels*4/5))
%            ord2 = ord2 + 1;
            %classIV_test(shuffle2(ord2),i) = 1;
%            classIV_test(shuffle2(ord2),:) = i-1;
%            testData(shuffle2(ord2),:) = IV(ord,:);   % 1000 x 100
%        else
%            ord3 = ord3 + 1;
%            classIV_valid(shuffle3(ord3),:) = i-1;
%            validData(shuffle3(ord3),:) = IV(ord,:);   % 1000 x 100
%        end
%    end  
%    i
%end



%csvwrite('trainData.csv',trainData)
%csvwrite('testData.csv',testData)
%csvwrite('class.csv',classIV)