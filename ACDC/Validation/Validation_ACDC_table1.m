set(0,'defaultTextInterpreter','tex')
set(0,'defaultAxesTickLabelInterpreter','tex')

% The original dataset can be found here: https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html
datafolder_orig = '/usr/local/micapollo01/MIC/DATA/STAFF/stilbo0/data/Data/ACDC/training';

% Tools to load and save .nii, download from: https://nl.mathworks.com/matlabcentral/fileexchange/8797-tools-for-nifti-and-analyze-image
addpath('/usr/local/micapollo01/MIC/DATA/STAFF/stilbo0/data/Software/NIfTI_20140122') 

main_dir = '/usr/local/micapollo01/MIC/DATA/STAFF/stilbo0/data/MICCAI_2022_scripts';
addpath('/usr/local/micapollo01/MIC/DATA/STAFF/stilbo0/data/Software/NIfTI_20140122')

resultfolder_main = fullfile(main_dir,'Trained_models');
datafolder = fullfile(main_dir,'Preprocessed_images');

%% Compare image-based and batch-based: results for MICCAI paper
resultSeries = {'record_0/Validation/full_image_DSC_Round_0_Fold_%i',... 
                'record_0/Validation/full_batch_DSC_Round_0_Fold_%i',... 
                'record_0/Validation/full_image_eps_DSC_Round_0_Fold_%i',...  
                'record_0/Validation/partial_image_DSC_Round_0_Fold_%i',... 
                'record_0/Validation/partial_batch_DSC_Round_0_Fold_%i',...
                'record_0/Validation/partial_image_eps_DSC_Round_0_Fold_%i',...
                };

tmp = load(fullfile(datafolder,'folds.mat'));
folds_images = tmp.folds_images;

% Validation
DSC = cell(length(resultSeries),1);
vol_gt = cell(length(resultSeries),1);
vol_result = cell(length(resultSeries),1);

for r = 1:length(resultSeries)
    % Validation
    DSC{r} = nan(200,3);
    vol_gt{r} = nan(200,3);
    vol_result{r} = nan(200,3);
end

for r = 1:length(resultSeries)
    
    for f =1:5
        idx = find(folds_images==f); % validation set
        for q = idx
            q
            % Load mask
            imgname = fullfile(datafolder,sprintf('ACDC_%04d_mask.nii.gz',q));
            img = load_nii(imgname);
            mask = img.img;
            ps = img.hdr.dime.pixdim(2:4);

            repeats = find(mask(96,96,:)==0);
            pos = find(diff(repeats)>1);
            repeats = repeats(pos:pos+1);

            % Load original gt
            patient = mod(q-1,100)+1;
            patientdir = fullfile(datafolder_orig,sprintf('patient%03d',patient));
            fid = fopen(fullfile(patientdir,'Info.cfg'));
            info = textscan(fid,'%s');
            ED = str2num(info{1}{2});
            ES = str2num(info{1}{4});
            if q>100, time = ES; else, time = ED; end
            
            imgname = fullfile(patientdir,sprintf('patient%03d_frame%02d_gt.nii.gz',patient,time));
            img = load_nii(imgname);
            gt_orig = flip(flip(flip(permute(img.img,[2 1 3]),3),2)*1);
            ps_orig = img.original.hdr.dime.pixdim(2:4);

            imgname = fullfile(patientdir,sprintf('patient%03d_frame%02d.nii.gz',patient,time));
            img = load_nii(imgname);
            I = flip(flip(flip(permute(img.img,[2 1 3]),3),2)*1);

            % Load result
            imgname = fullfile(resultfolder_main,sprintf('case_%04d',q),sprintf(resultSeries{r},f-1),'full_val__s0__b0__x.nii.gz');
            if ~exist(imgname,'file')
                imgname = strrep(imgname,'Round_1','Round_0');
                if ~exist(imgname,'file')
                    imgname = strrep(imgname,'MICCAI_experiments','DVN2');
                    if ~exist(imgname,'file')
                        imgname = fullfile(resultfolder_main,sprintf('case_%04d_0',q),sprintf(resultSeries{r},f-1),'full_val__s0__b0__x.nii.gz');
                        if ~exist(imgname,'file') 
                            imgname = strrep(imgname,'Round_1','Round_0');
                        end
                    end
                end
            end
            img = load_untouch_nii(imgname);
            result = img.img;

            % calculate transformation matrix
            Shape = size(result);
            rot_origin = (size(gt_orig)+1)/2;
            S = [ps(2)/ps_orig(2) 0 0 0; 0 ps(1)/ps_orig(1) 0 0; 0 0 ps(3)/ps_orig(3) 0; 0 0 0 1];
            T = [1 0 0 rot_origin(2) ; 0 1 0 rot_origin(1) ;0 0 1 rot_origin(3) ;0 0 0 1];
            T_ = [1 0 0 -(Shape(2)+1)/2;0 1 0 -(Shape(1)+1)/2; 0 0 1 -(Shape(3)+1)/2; 0 0 0 1]; 

            Shape_restore = size(gt_orig);
            [x,y,z]=meshgrid(1:Shape_restore(2),1:Shape_restore(1),1:Shape_restore(3));
            x = x(:)'; y = y(:)'; z = z(:)';
            xyz_restore = pinv(T*S*T_)*double([x; y; z; ones(size(x))]); xyz_restore = round(xyz_restore,10);

            result_restore = nan([Shape_restore 4]);
            result(repmat(mask,[1,1,1,4])==0)=nan;
            result(:,:,repeats(1),:) = result(:,:,repeats(1)+1,:);
            result(:,:,repeats(2),:) = result(:,:,repeats(2)-1,:);
            for k = 1 : 4
                result_restore(:,:,:,k)=interp3(double(result(:,:,:,k)),reshape(xyz_restore(1,:),Shape_restore),reshape(xyz_restore(2,:),Shape_restore),reshape(xyz_restore(3,:),Shape_restore),'linear',NaN);
            end

            [~,result_restore_cat] = max(result_restore,[],4);
            result_restore_cat = result_restore_cat-1;

            % binary gt and result: LV cavity, myocardium, RV cavity
            gt_bin = nan([size(gt_orig) 3]);
            gt_bin(:,:,:,1) = double(gt_orig==3);
            gt_bin(:,:,:,2) = double(gt_orig==2);
            gt_bin(:,:,:,3) = double(gt_orig==1);
            result_bin = nan(size(gt_bin));
            result_bin(:,:,:,1) = double(result_restore_cat==1);
            result_bin(:,:,:,2) = double(result_restore_cat==2);
            result_bin(:,:,:,3) = double(result_restore_cat==3);

            DSC{r}(q,:) = 2*sum((gt_bin.*result_bin),[1,2,3])./(sum(gt_bin,[1,2,3])+sum(result_bin,[1,2,3]));
            
            [x,y,z]=meshgrid(1:size(gt_bin,2),1:size(gt_bin,1),1:size(gt_bin,3));
            for k = 1:3
                vol_gt{r}(q,k) = nansum(gt_bin(:,:,:,k),[1 2 3])*prod(ps_orig)/10^3;
                vol_result{r}(q,k) = nansum(result_bin(:,:,:,k),[1 2 3])*prod(ps_orig)/10^3;
            end
        end
    end
end

thr = 20;
for r = 1:length(resultSeries)
    ES_as_ES{r} = nnz(find(vol_result{r}(101:200,2)<=thr))/nnz(find(~isnan(vol_result{r}(101:200,2)))); % true positive rate for ES detection = sensitivity
    ES_as_ED{r} = nnz(find(vol_result{r}(101:200,2)>thr))/nnz(find(~isnan(vol_result{r}(101:200,2))));  
    
    ED_as_ES{r} = nnz(find(vol_result{r}(1:100,2)<=thr))/nnz(find(~isnan(vol_result{r}(1:100,2)))); 
    ED_as_ED{r} = nnz(find(vol_result{r}(1:100,2)>thr))/nnz(find(~isnan(vol_result{r}(1:100,2)))); % true negative rate for ES dectection = specificity
    
    precision{r} = nnz(find(vol_result{r}(101:200,2)<=thr))/nnz(find(vol_result{r}(1:200,2)<=5));% precision for ES detection
end

colors_ = {'k--','b--','r--'};
sensitivity_all = zeros(1001,3);
specificity_all = zeros(1001,3);
figure
for r = 4:length(resultSeries)
    fractions = 0:max(vol_result{r}(:,2))/1000:max(vol_result{r}(:,2));
    sensitivity=zeros(size(fractions));
    specificity = zeros(size(fractions));
    for idx=1:length(fractions)
        f = fractions(idx);
        sensitivity(idx) = nnz(find(vol_result{r}(101:200,2)<f))/nnz(find(~isnan(vol_result{r}(101:200,2))));
        specificity(idx) =  nnz(find(vol_result{r}(1:100,2)>=f))/nnz(find(~isnan(vol_result{r}(1:100,2))));
    end
    sensitivity_all(:,r-3) = sensitivity;
    specificity_all(:,r-3) = specificity;
    p=(1-specificity);
    q=sensitivity;
    AOC_bin{r} = sum(((p(2:end)-p(1:end-1)).*(q(2:end)-q(1:end-1)))/2+(1-p(2:end)).*(q(2:end)-q(1:end-1)));
    plot(1-sensitivity_all(:,r-3),specificity_all(:,r-3),colors_{r-3}), hold on
    ylabel('Sensitivity')
    xlabel('1-Specificity')
    title('MYO_{ED} classification in ACDC')
    legend('I, B=2: 0.43','BI, B=2: 0.92','I, B=2: 0.94')
end

% save(fullfile(resultfolder_main,'sensitivity_specificity.mat'),'sensitivity_all','specificity_all')

experiments = {'Full I','Full BI', 'Full Ie', 'Partial I', 'Partial BI', 'Partial Ie'};
for r = 1:length(resultSeries)
    m_ED = round(nanmean(DSC{r}(1:100,:)),2);
    m_ES = round(nanmean(DSC{r}(101:200,:)),2);
    string_ = sprintf('experiment %s ',experiments{r});
    for k = 2%1:3
        string_ = strcat(string_,sprintf('& ED %0.2f',m_ED(k)));
    end
    for k = 2%1:3
        string_ = strcat(string_,sprintf('& ES %0.2f',m_ES(k)));
    end
    string_
end

for r = 1:length(resultSeries)
    dV = vol_result{r}-vol_gt{r};
    m_ED = round(nanmean(dV(1:100,:)),0);
    m_ES = round(nanmean(dV(101:200,:)),0);
    string_ = sprintf('experiment %s ',experiments{r});
    for k = 2
        string_ = strcat(string_,sprintf('& ED %i',m_ED(k)));
    end
    for k = 2
        string_ = strcat(string_,sprintf('& ES %i',m_ES(k)));
    end
    string_
end

% calculate statistical significances of difference myocardium segmentation
% between full and partial labeling
rank_labels_DSC = cell(3,2);
p_values_labels_DSC = cell(3,2);
ranks_tot_labels_DSC = cell(3,2);
k = 2; %MYO
for p=1:2 % p = 1 -> (ED), p = 2 -> (ES)
    for r = 1:3
        [rank_labels_DSC{r,p},p_values_labels_DSC{r,p},ranks_tot_labels_DSC{r,p}] = get_ranks({DSC{r}(1+100*(p-1):100+100*(p-1),k) ,DSC{r+3}(1+100*(p-1):100+100*(p-1),k)},2,'descend',1);
    end
end

rank_labels_vol = cell(3,2);
p_values_labels_vol = cell(3,2);
ranks_tot_labels_vol = cell(3,2);
for p=1:2
    for r =1:3
        [rank_labels_vol{r,p},p_values_labels_vol{r,p},ranks_tot_labels_vol{r,p}] = get_ranks({abs(vol_gt{r}(1+100*(p-1):100+100*(p-1),k)-vol_result{r}(1+100*(p-1):100+100*(p-1),k)) ,abs(vol_gt{r+3}(1+100*(p-1):100+100*(p-1),k)-vol_result{r+3}(1+100*(p-1):100+100*(p-1),k))},2,'ascend',1);
    end
end
