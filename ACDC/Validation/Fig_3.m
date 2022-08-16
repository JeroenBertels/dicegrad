% set(0,'defaultTextInterpreter','tex')
% set(0,'defaultAxesTickLabelInterpreter','tex')

addpath('/usr/local/micapollo01/MIC/DATA/STAFF/stilbo0/data/MatlabCode/DocumentedCode/SegmentationT1map')
addpath('/usr/local/micapollo01/MIC/DATA/STAFF/stilbo0/data/Software/export_fig')
addpath('/usr/local/micapollo01/MIC/DATA/STAFF/stilbo0/data/Software/NIfTI_20140122')
addpath('/usr/local/micapollo01/MIC/DATA/STAFF/stilbo0/data/Software/SliceBrowser')
addpath('/usr/local/micapollo01/MIC/DATA/STAFF/stilbo0/data/Software/3DMeshToBinaryImage/3DMeshToBinaryImage')
addpath('/usr/local/micapollo01/MIC/DATA/STAFF/stilbo0/data/MatlabCode/CNNestimateShapeModel/FullPaper/')

resultfolder_main = '/usr/local/micapollo01/MIC/DATA/STAFF/stilbo0/tmp/Experiments/TOF/MICCAI_experiments/runs/run_1';
datafolder = '/usr/local/micapollo01/MIC/DATA/STAFF/stilbo0/tmp/Data/ACDC_3d_distancemaps';
datafolder_orig = '/usr/local/micapollo01/MIC/DATA/STAFF/stilbo0/data/Data/ACDC/training';


%% images of fig. 3
resultSeries = {'record_0/Validation/noMyoES_imageDSC_noBG_earlyStopping_40_80_Round_0_Fold_%i',... 
                'record_0/Validation/noMyoES_batchDSC_noBG_earlyStopping_40_80_Round_1_Fold_%i',...
                'record_0/Validation/noMyoES_imageDSC_noBG_earlyStopping_40_80_eps_meanVolAlpha_frac_0k75_smallEpsLVRV_Round_1_Fold_%i',...%not with small epsLVRV
                };

tmp = load(fullfile(datafolder,'folds.mat'));
folds_images = tmp.folds_images;

for r = 1:length(resultSeries)
    for f =4
        idx = [99 199];
        for q = [99 199]
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

            patient = mod(q-1,100)+1; % select cases in paper based on 10%,50%,90% of myocardial DSC ES
            phase = ceil(q/100);
            midslice = round(size(I,3)/2);
            segmentation = double(gt_orig>0);
            segmentation_midslice =segmentation(:,:,midslice);
            [x,y]=meshgrid(1:size(segmentation_midslice,2),1:size(segmentation_midslice,1));
            seg_mid_weighted_x = segmentation_midslice.*x;
            seg_mid_weighted_y = segmentation_midslice.*y;
            midpoint = round([sum(seg_mid_weighted_y(:)) sum(seg_mid_weighted_x(:))]/sum(segmentation_midslice(:))); 
            ROI = [midpoint(1)-50 midpoint(1)+50 midpoint(2)-50 midpoint(2)+50];
            qq = quantile(I(:),100);
            I(I<qq(1)) = qq(1);
            I(I>qq(100)) = qq(100);
            colors = {'r','g','c'};
            labels={'GT','I','IB','Ie'};
            fig = figure;
            imshow(I(ROI(1):ROI(2),ROI(3):ROI(4),midslice),[])
            fig.Position = [680 691 355 275];
            result_bin(:,:,:,2) = result_bin(:,:,:,1)+result_bin(:,:,:,2);
            for k = 3:-1:1
                contour_ = contourc(result_bin(ROI(1):ROI(2),ROI(3):ROI(4),midslice,k),1);
                hold on, plot(contour_(1,2:end),contour_(2,2:end),colors{k},'Linewidth',1.5)
            end
            pause(0.5)
            if phase==1
                %title(labels{r+1},'Fontweight','normal','Interpreter','Latex');
                fig.Color = 'White';
                %export_fig(fullfile(resultfolder_main,sprintf('%s_ED.pdf',labels{r+1})),'-r120')
            else
                fig.Color = 'White';
                %export_fig(fullfile(resultfolder_main,sprintf('%s_ES.pdf',labels{r+1})),'-r120')
            end

            if r==1
                fig = figure;
                imshow(I(ROI(1):ROI(2),ROI(3):ROI(4),midslice),[])
                fig.Position = [680 691 355 275];
                %plots_{phase,1} = subplot(2,4,(phase-1)*4+1);imshow(I(ROI(1):ROI(2),ROI(3):ROI(4),midslice),[])
                gt_bin(:,:,:,2) = gt_bin(:,:,:,1)+gt_bin(:,:,:,2);
                for k = 3:-1:1
                    if ~(k==2 && phase==2)
                        contour_ = contourc(gt_bin(ROI(1):ROI(2),ROI(3):ROI(4),midslice,k),1);
                        hold on, plot(contour_(1,2:end),contour_(2,2:end),colors{k},'Linewidth',1.5)
                    end
                end
                pause(0.5)
                if phase==1
%                     lb = ylabel('ED','FontWeight','normal','Interpreter','Latex');
%                     title(labels{1},'FontWeight','normal','Interpreter','Latex');
                    fig.Color = 'White';
                    %export_fig(fullfile(resultfolder_main,'GT_ED.pdf'),'-r120')
                else
                    %lb = ylabel('ES','Interpreter','Latex');
                    fig.Color = 'White';
                    %export_fig(fullfile(resultfolder_main,'GT_ES.pdf'),'-r120')
                end
                lb.Position(1)=-5;
            end
        end
    end

end