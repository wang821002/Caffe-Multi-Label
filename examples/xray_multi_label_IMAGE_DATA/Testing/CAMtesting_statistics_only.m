% Sample code to generate class activation map from 10 crops of activations
% Bolei Zhou, March 15, 2016
% for the online prediction, make sure you have complied matcaffe

clear
addpath('/home/osboxes/caffe-exp-A2L1/matlab/caffe');
load('categories.mat');

imageDir = '/home/osboxes/caffe-exp-A2L1/examples/xray_multi_label_IMAGE_DATA/images/';
outputDir = 'CAM-Results/';
% imgID = 2; % 1 or 2
% img = imread(['img' num2str(imgID) '.jpg']);
% img = imresize(img, [256 256]);
online = 1; % whether extract features online or load pre-extracted features

trainedModelDir = ['../trained_model/' 'aaa_trained_resnet_10000n_8_hinge_AVE_iter4_fast/'];
net_weights = [trainedModelDir 'CAM_resnet_train_8_iter_12000.caffemodel'];
net_model = ['deploy_CAM_resnet_Xray_AVE_noSigmoid.prototxt'];
net = caffe.Net(net_model, net_weights, 'test');
caffe.set_mode_gpu();
gpu_id =0;  % we will use the first gpu in this demo
caffe.set_device(gpu_id);
% net = caffe('init', net_model, net_weights)
% caffe('set_mode_gpu');
% caffe('set_phase_test');

% weights_LR = net.params('CAM_fc_xray',1).get_data();% get the softmax layer of the network

% f=figure;
% imageList = dir(imageDir); imageList=imageList(3:end);
dataset = '1000';
%% for training and validation images
load(['label_list_' dataset '.mat']);
% load('label_list_10000n.mat');
imageLabelList = [Train_label_all; Val_label_all]; 
imageList = imageLabelList(:,1);

labelNum=8;

classTotalNum = sum(cell2mat(imageLabelList(:,2:labelNum+1))>0,1);

ROC_T = [0:0.01:0.99,0.991:0.001:1];
T_num = size(ROC_T,2);
hamming_distance_temp = zeros(T_num,labelNum);
classTPFPTNFNNum = zeros(T_num,labelNum,4);
confMat_train = zeros(T_num, labelNum, labelNum);
% for i=1:length(imageList)
%     i
%     imagePath = imageList{i,1};
%     if exist([outputDir 'r*' imagePath(end-14:end) ],'file'); continue;end
%     img = imread([imageDir imagePath(end-14:end)]);
%     if size(img,3)~=3
%         img= cat(3,img,img,img);
%     end
% 
%     % load the CAM model and extract features      
% 
%     scores = net.forward({prepare_image(img)});% extract conv features online
% %     activation_lastconv = net.blobs('CAM_conv_Xray').get_data();
% %     scores = scores{1};
%     
%     for i_t = 1:T_num
%     % compute Hamming distance
%         prediction = mean(scores{1},2) >= ROC_T(i_t);
%         gt = (cell2mat(imageLabelList(i,2:labelNum+1)) > 0)';
%         
%         for ii = 1:labelNum
%             if gt(ii, 1) == 1
%                 confMat_train(i_t,ii,:) = reshape(confMat_train(i_t,ii,:),8,1) + prediction;
%             end
%         end
% 
%         % compute TP FP TN FN
%         for ii= 1:labelNum
%             if prediction(ii) == 1 && gt(ii) == 1
%                 classTPFPTNFNNum(i_t,ii,1) = classTPFPTNFNNum(i_t,ii,1) + 1;
%             elseif prediction(ii) == 1 && gt(ii) == 0
%                 classTPFPTNFNNum(i_t,ii,2) = classTPFPTNFNNum(i_t,ii,2) + 1;
%             elseif prediction(ii) == 0 && gt(ii) == 0
%                 classTPFPTNFNNum(i_t,ii,3) = classTPFPTNFNNum(i_t,ii,3) + 1;
%             elseif prediction(ii) == 0 && gt(ii) == 1
%                 classTPFPTNFNNum(i_t,ii,4) = classTPFPTNFNNum(i_t,ii,4) + 1;
%             end
%             hamming_distance_temp(i_t,ii) = hamming_distance_temp(i_t,ii) + (prediction(ii) == gt(ii));
%         end
%     end
% %     %% Class Activation Mapping
% %     GT_index = find(gt(1:8,1));
% %     topNum = length(GT_index); % generate heatmap for prediction results where gt == 1
% % %     scoresMean = mean(scores,2);
% % %     [value_category, IDX_category] = sort(scoresMean,'descend');
% % %     [curCAMmapAll] = returnCAMmap(activation_lastconv, weights_LR(:,IDX_category(1:topNum)));
% % %     GT_index = find(gt);
% %     [curCAMmapAll] = returnCAMmap(activation_lastconv, weights_LR(:,GT_index));
% % 
% % 
% %     
% %     
% % 
% %     for j=1:topNum
% %         curCAMmap_crops = squeeze(curCAMmapAll(:,:,j,:));
% %         curCAMmapLarge_crops = imresize(curCAMmap_crops,[512 512]);
% %         curCAMLarge = mergeTenCrop(curCAMmapLarge_crops);
% % %         curCAMLarge = curCAMmapLarge_crops;
% %         curHeatMap = imresize(im2double(curCAMLarge),[512 512]);
% %         curHeatMap = im2double(curHeatMap);
% % 
% %         curHeatMap = map2jpg(curHeatMap,[], 'jet');
% %         curHeatMap = im2double(img)*0.3+curHeatMap*0.6;
% %         curResult = im2double(img);
% %         curResult = [curResult ones(size(curHeatMap,1),8,3) curHeatMap];
% % %         curPrediction = [curPrediction ' --top'  num2str(j) ':' categories{IDX_category(j)}];
% %         curPrediction = '';
% %         curPrediction = [curPrediction ' --Groundtruth Class '  num2str(GT_index(j)) ':' categories{GT_index(j)}];
% %         imshow(curResult);title(curPrediction);
% %         ff = getframe;
% %         imwrite(ff.cdata,[outputDir imagePath(end-14:end-4) '_c_' num2str(GT_index(j)) '.jpg']);
% %     end
% 
%     
%     
%     
% end
% sensitivity_train = zeros(T_num,labelNum);
% specificity_train = zeros(T_num,labelNum);
% precision_train = zeros(T_num,labelNum);
% F1_train = zeros(T_num,labelNum);
% hamming_distance_train = zeros(T_num,labelNum);
% hamming_distance_all_train = zeros(T_num);
% for i_t=1:T_num
%     sensitivity_train(i_t,:) = classTPFPTNFNNum(i_t,:,1) ./ classTotalNum;
%     specificity_train(i_t,:) = classTPFPTNFNNum(i_t,:,3) ./ (size(imageList,1) - classTotalNum);
%     precision_train(i_t,:) = classTPFPTNFNNum(i_t,:,1) ./ (classTPFPTNFNNum(i_t,:,1)+classTPFPTNFNNum(i_t,:,2));
%     F1_train(i_t,:) = 2*classTPFPTNFNNum(i_t,:,1) ./ (2*classTPFPTNFNNum(i_t,:,1) + classTPFPTNFNNum(i_t,:,2) + classTPFPTNFNNum(i_t,:,4));
%     hamming_distance_train(i_t,:) = hamming_distance_temp(i_t,:) ./ classTotalNum;
%     hamming_distance_all_train(i_t) = sum(hamming_distance_temp(i_t,:)) /labelNum / (size(imageList,1));
% end
% colors = {'b','k','r','g','b--','k--','r--','g--'};
% figure;
% for i=1:labelNum
%     plot(1-specificity_train(2:end-1,i),sensitivity_train(2:end-1,i),colors{i});
%     hold on;
% end
% hold off;
% plot(0:0.1:1,0:0.1:1,'k:');
% legend('Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltrate', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax');
% 
% %% compute AUC
% AUC_train = zeros(1,8);
% for i=1:size(ROC_T,2)-1
%    sen_temp =  sensitivity_train(i+1,:);
%    spec_temp_1 = specificity_train(i,:);
%    spec_temp_2 = specificity_train(i+1,:);
%    spec_diff_temp = spec_temp_2-spec_temp_1;
%    AUC_train = AUC_train + sen_temp .* spec_diff_temp;    
% end
% 


%% for testing images

imageLabelList = Test_label_all; 
imageList = imageLabelList(:,1);

labelNum=8;

classTotalNum = sum(cell2mat(imageLabelList(:,2:labelNum+1))>0,1);

hamming_distance_temp = zeros(T_num,labelNum);
classTPFPTNFNNum = zeros(T_num,labelNum,4);
confMat_test = zeros(T_num, labelNum, labelNum);
for i=1:length(imageList)
    i
    imagePath = imageList{i,1};
    if exist([outputDir 'r*' imagePath(end-14:end) ],'file'); continue;end
    img = imread([imageDir imagePath(end-14:end)]);
    if size(img,3)~=3
        img= cat(3,img,img,img);
    end

    % load the CAM model and extract features      
    img_size = 524;
    img600 = imresize(img, [img_size img_size]);
    scores = net.forward({prepare_image(img600,img_size)});% extract conv features online
%     activation_lastconv = net.blobs('CAM_conv_Xray').get_data();
%     scores = scores{1};
    
    for i_t = 1:T_num
    % compute Hamming distance
        prediction = mean(scores{1},2) >= ROC_T(i_t);
        gt = (cell2mat(imageLabelList(i,2:labelNum+1)) > 0)';
        
        for ii = 1:labelNum
            if gt(ii, 1) == 1
                confMat_test(i_t,ii,:) = reshape(confMat_test(i_t,ii,:),8,1) + prediction;
            end
        end
        % compute TP FP TN FN
        for ii= 1:labelNum
            if prediction(ii) == 1 && gt(ii) == 1
                classTPFPTNFNNum(i_t,ii,1) = classTPFPTNFNNum(i_t,ii,1) + 1;
            elseif prediction(ii) == 1 && gt(ii) == 0
                classTPFPTNFNNum(i_t,ii,2) = classTPFPTNFNNum(i_t,ii,2) + 1;
            elseif prediction(ii) == 0 && gt(ii) == 0
                classTPFPTNFNNum(i_t,ii,3) = classTPFPTNFNNum(i_t,ii,3) + 1;
            elseif prediction(ii) == 0 && gt(ii) == 1
                classTPFPTNFNNum(i_t,ii,4) = classTPFPTNFNNum(i_t,ii,4) + 1;
            end
            hamming_distance_temp(i_t,ii) = hamming_distance_temp(i_t,ii) + (prediction(ii) == gt(ii));
        end
    end
%     %% Class Activation Mapping
%     GT_index = find(gt(1:8,1));
%     topNum = length(GT_index); % generate heatmap for prediction results where gt == 1
% %     scoresMean = mean(scores,2);
% %     [value_category, IDX_category] = sort(scoresMean,'descend');
% %     [curCAMmapAll] = returnCAMmap(activation_lastconv, weights_LR(:,IDX_category(1:topNum)));
% %     GT_index = find(gt);
%     [curCAMmapAll] = returnCAMmap(activation_lastconv, weights_LR(:,GT_index));
% 
% 
%     
%     
% 
%     for j=1:topNum
%         curCAMmap_crops = squeeze(curCAMmapAll(:,:,j,:));
%         curCAMmapLarge_crops = imresize(curCAMmap_crops,[512 512]);
%         curCAMLarge = mergeTenCrop(curCAMmapLarge_crops);
% %         curCAMLarge = curCAMmapLarge_crops;
%         curHeatMap = imresize(im2double(curCAMLarge),[512 512]);
%         curHeatMap = im2double(curHeatMap);
% 
%         curHeatMap = map2jpg(curHeatMap,[], 'jet');
%         curHeatMap = im2double(img)*0.3+curHeatMap*0.6;
%         curResult = im2double(img);
%         curResult = [curResult ones(size(curHeatMap,1),8,3) curHeatMap];
% %         curPrediction = [curPrediction ' --top'  num2str(j) ':' categories{IDX_category(j)}];
%         curPrediction = '';
%         curPrediction = [curPrediction ' --Groundtruth Class '  num2str(GT_index(j)) ':' categories{GT_index(j)}];
%         imshow(curResult);title(curPrediction);
%         ff = getframe;
%         imwrite(ff.cdata,[outputDir imagePath(end-14:end-4) '_c_' num2str(GT_index(j)) '.jpg']);
%     end

    
    
    
end
sensitivity_test = zeros(T_num,labelNum);
specificity_test = zeros(T_num,labelNum);
precision_test = zeros(T_num,labelNum);
F1_test = zeros(T_num,labelNum);
hamming_distance_test = zeros(T_num,labelNum);
hamming_distance_all_test = zeros(T_num);
for i_t=1:T_num
    sensitivity_test(i_t,:) = classTPFPTNFNNum(i_t,:,1) ./ classTotalNum;
    specificity_test(i_t,:) = classTPFPTNFNNum(i_t,:,3) ./ (size(imageList,1) - classTotalNum);
    precision_test(i_t,:) = classTPFPTNFNNum(i_t,:,1) ./ (classTPFPTNFNNum(i_t,:,1)+classTPFPTNFNNum(i_t,:,2));
    F1_test(i_t,:) = 2*classTPFPTNFNNum(i_t,:,1) ./ (2*classTPFPTNFNNum(i_t,:,1) + classTPFPTNFNNum(i_t,:,2) + classTPFPTNFNNum(i_t,:,4));
    hamming_distance_test(i_t,:) = hamming_distance_temp(i_t,:) ./ classTotalNum;
    hamming_distance_all_test(i_t) = sum(hamming_distance_temp(i_t,:)) /labelNum / (size(imageList,1));
end

colors = {'b','k','r','g','b--','k--','r--','g--'};
figure;
for i=1:labelNum
    plot([1;1-specificity_test(2:end-1,i) ],[1;sensitivity_test(2:end-1,i) ],colors{i});
    hold on;
end

plot(0:0.1:1,0:0.1:1,'k:');
legend('Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltrate', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax');
hold off;
%% compute AUC
AUC_test = zeros(1,8);
for i=1:size(ROC_T,2)-1
   sen_temp =  sensitivity_test(i+1,:);
   spec_temp_1 = specificity_test(i,:);
   spec_temp_2 = specificity_test(i+1,:);
   spec_diff_temp = spec_temp_2-spec_temp_1;
   AUC_test = AUC_test + sen_temp .* spec_diff_temp;    
end

    
save([net_weights '_' dataset '_NewResults.mat']);

caffe.reset_all();

