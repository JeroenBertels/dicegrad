% Script for preprocessing ACDC dataset

% The original dataset can be found here: https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html
datafolder = '/usr/local/micapollo01/MIC/DATA/STAFF/stilbo0/data/Data/ACDC/training';

% Tools to load and save .nii, download from: https://nl.mathworks.com/matlabcentral/fileexchange/8797-tools-for-nifti-and-analyze-image
addpath('/usr/local/micapollo01/MIC/DATA/STAFF/stilbo0/data/Software/NIfTI_20140122') 

% Specify directory
main_dir = '/usr/local/micapollo01/MIC/DATA/STAFF/stilbo0/data/MICCAI_2022_scripts';

writedir = fullfile(main_dir,'Preprocessed_images');
if ~exist(writedir,'dir')
    mkdir(writedir)
end

h = dir(fullfile(datafolder,'patient*'));
h = h([h.isdir]);

warning('off')

ref_spacing = [1.5625 1.5625 2.5];
ref_size = [192 192 48];

folds_images = repmat(1:5,[1 40]);
save(fullfile(writedir,'folds.mat'))

sizes = nan(200,3);

idx = 1;
for i = 1:length(h)
    display(sprintf('Patient_%i',i))
    % Full 4D scan: read to know pixelspacing
    file = fullfile(h(i).folder,h(i).name,strcat(h(i).name,'_4d.nii.gz'));
    info_4D = load_nii(file);
    I4D = info_4D.img;
    pixsp = info_4D.original.hdr.dime.pixdim([3 2]);
    if pixsp(1) ~= pixsp(2)
        display('x and y direction do not have same dimensions')
    end
    slicethickness = info_4D.original.hdr.dime.pixdim(4);
    
    fid = fopen(fullfile(h(i).folder,h(i).name,'Info.cfg'));
    info = textscan(fid,'%s');
    ED = str2num(info{1}{2});
    ES = str2num(info{1}{4});
    Group = info{1}{6};
        
    % gt's and corresponding mr 
    % 1 = RV
    % 2 = Myo
    % 3 = LV
    for t = [ED ES]
        
        if t==ED, idx_ = idx; else, idx_ = idx+100; end % make sure ED and ES of same patient are in the same fold

        mr_file = fullfile(h(i).folder,h(i).name,strcat(h(i).name,sprintf('_frame%02d.nii.gz',t)));
        img = load_nii(mr_file);
        I = flip(flip(flip(permute(img.img,[2 1 3]),3),2)*1);

        pix_spacing = round([pixsp(1) pixsp(2) slicethickness],4);
        Shape = ref_size;
        [x,y,z]=meshgrid(1:Shape(2),1:Shape(1),1:Shape(3));
        x = x(:)'; y = y(:)'; z = z(:)';
        rot_origin = (size(I)+1)/2;
        S = [ref_spacing(2)/pix_spacing(2) 0 0 0; 0 ref_spacing(1)/pix_spacing(1) 0 0; 0 0 ref_spacing(3)/pix_spacing(3) 0; 0 0 0 1];
        T = [1 0 0 rot_origin(2) ; 0 1 0 rot_origin(1) ;0 0 1 rot_origin(3) ;0 0 0 1];
        T_ = [1 0 0 -(Shape(2)+1)/2;0 1 0 -(Shape(1)+1)/2; 0 0 1 -(Shape(3)+1)/2; 0 0 0 1]; 

        xyz_new = T*S*T_*double([x; y; z; ones(size(x))]); xyz_new = round(xyz_new,10);
        image_new=interp3(double(I),reshape(xyz_new(1,:),Shape),reshape(xyz_new(2,:),Shape),reshape(xyz_new(3,:),Shape),'linear',NaN);

        gt_file = strrep(mr_file,'.nii.gz','_gt.nii.gz');
        gt = load_nii(gt_file);
        gt = flip(flip(flip(permute(gt.img,[2 1 3]),3),2)*1);

        ns = size(gt,3);

        % get separate structures (binary needed for negative/positive; smooth needed for distance)
        endo = double(gt==3);
        epi = double(gt>1);
        rv = double(gt==1);
        myo = epi-endo;
        heart = double(gt>0);

        % create distance maps
        [x,y,z]=meshgrid(1:size(gt,2),1:size(gt,1),1:size(gt,3));

        surface_endo = isosurface(x,y,z,endo,1/2);
        distance_endo = reshape(min(pdist2(surface_endo.vertices.*pix_spacing,[x(:) y(:) z(:)].*pix_spacing)),size(gt));
        distance_endo = distance_endo .* (double(endo)-0.5)*(-2);

        surface_epi = isosurface(x,y,z,epi,1/2);
        surface_myo.vertices = [surface_endo.vertices ; surface_epi.vertices];
        distance_myo = reshape(min(pdist2(surface_myo.vertices.*pix_spacing,[x(:) y(:) z(:)].*pix_spacing)),size(gt));
        distance_myo = distance_myo .* (double(myo)-0.5)*(-2);

        surface_rv = isosurface(x,y,z,rv,1/2);
        distance_rv = reshape(min(pdist2(surface_rv.vertices.*pix_spacing,[x(:) y(:) z(:)].*pix_spacing)),size(gt));
        distance_rv = distance_rv .* (double(rv)-0.5)*(-2);

        surface_heart = isosurface(x,y,z,heart,1/2);
        distance_heart = pdist2(surface_heart.vertices.*pix_spacing,[x(:) y(:) z(:)].*pix_spacing);
        distance_heart = min(distance_heart);
        distance_heart = reshape(distance_heart,size(gt));
        distance_heart = distance_heart .* (double(heart)-0.5)*(2); % background

        % resample distance maps
        distance_endo_new=interp3(double(distance_endo),reshape(xyz_new(1,:),Shape),reshape(xyz_new(2,:),Shape),reshape(xyz_new(3,:),Shape),'linear',NaN);
        distance_myo_new=interp3(double(distance_myo),reshape(xyz_new(1,:),Shape),reshape(xyz_new(2,:),Shape),reshape(xyz_new(3,:),Shape),'linear',NaN);
        distance_rv_new=interp3(double(distance_rv),reshape(xyz_new(1,:),Shape),reshape(xyz_new(2,:),Shape),reshape(xyz_new(3,:),Shape),'linear',NaN);
        distance_heart_new=interp3(double(distance_heart),reshape(xyz_new(1,:),Shape),reshape(xyz_new(2,:),Shape),reshape(xyz_new(3,:),Shape),'linear',NaN);

        mask = ~isnan(image_new);
        q = quantile(image_new(:),100);
        image_new(image_new<q(1))=q(1);
        image_new(image_new>q(100))=q(100);
        image_new(~mask)=0;

        image_new = (image_new-mean(image_new(:)))/std(image_new(:));
        nii_temp = make_nii(image_new,ref_spacing,[0 0 0],16);
        save_nii(nii_temp,fullfile(writedir,sprintf('ACDC_%04d.nii.gz',idx_)));
        nii_temp = make_nii(double(mask),ref_spacing,[0 0 0],2);
        save_nii(nii_temp,fullfile(writedir,sprintf('ACDC_%04d_mask.nii.gz',idx_)));

        distance_tot_new = cat(4,distance_heart_new,distance_endo_new,distance_myo_new,distance_rv_new);
        distance_tot_new(~repmat(mask,[1 1 1 size(distance_tot_new,4)])) = 100;

        [~,k] = max(convert_sigmoid(distance_tot_new),[],4);
        gt_categorical = k-1;
        nii_temp = make_nii(gt_categorical,ref_spacing,[0 0 0],16);
        save_nii(nii_temp,fullfile(writedir,sprintf('ACDC_%04d_gt_categorical.nii.gz',idx_)));

        sizes(idx_,:) = size(image_new);
        
    end
    idx = idx+1;
end

function soft_binary = convert_sigmoid(distance)
    distance(distance>10)=10;
    distance(distance<-10)=-10;
    soft_binary = exp(-5*distance)./(1+exp(-5*distance));
end