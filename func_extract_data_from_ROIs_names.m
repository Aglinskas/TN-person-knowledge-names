function roi_data = func_extract_data_from_ROIs_names(roi_dir,spm_dir)
%roi_data = func_extract_data_from_ROIs(roi_dir,spm_dir,ntasks,nsubs)
addpath(genpath('/Users/aidasaglinskas/Documents/MATLAB/spm12/toolbox/marsbar/'));
%% Get Masks
masks = [];
extract_voxel_data = 1;
masks.dir = roi_dir;
temp = dir([masks.dir 'R*.mat']);

masks.mat_files = {temp.name}'; clear temp;
ch = cellfun(@isempty,cellfun(@(x) strfind(x,'_labels'),masks.mat_files,'UniformOutput',0));
masks.mat_files = masks.mat_files(ch);
masks.nii_files = strrep(masks.mat_files,'.mat','.nii');

    masks.lbls = masks.nii_files;
    masks.lbls = strrep(masks.lbls,'ROI_','');
    masks.lbls = strrep(masks.lbls,'.nii','');
    masks.lbls = strrep(masks.lbls,'.mat','');
    
% Add paths to files    
masks.nii_files = fullfile(masks.dir,masks.nii_files);
masks.mat_files = fullfile(masks.dir,masks.mat_files);
disp(masks.lbls);
disp([num2str(length(masks.lbls)) ' ROIs found']);
%% Get ROI Mean Data
%loadMR
%spm_dir = '/Users/aidasaglinskas/Google Drive/Aidas:  Summaries & Analyses (WP 1.4)/Data_faces/Group_Analysis/';
spm_path = [spm_dir 'SPM.mat'];
load(spm_path);

tempa = cellfun(@(x) strsplit(x,filesep),SPM.xY.P,'UniformOutput',0);
tempb = cellfun(@(x) x{end},tempa,'UniformOutput',0);
nconds = length(unique(tempb));
nsubs = SPM.nscan / nconds;
disp(sprintf('%d Conds',nconds))
disp(sprintf('%d Subjects',nsubs))

D  = mardo(spm_path);% Make marsbar design object
roi_data = [];
for r_ind = 1:length(masks.mat_files)
ROI_fn = masks.mat_files{r_ind};
R  = maroi(ROI_fn);% Make marsbar ROI object
Y  = get_marsy(R, D, 'mean'); % Fetch data into marsbar data object
y = summary_data(Y); % Take 
r = reshape(y,nconds,nsubs);
roi_data.mat(r_ind,:,:) = r;
end

roi_data.lbls = masks.lbls;
ofn = '/Users/aidasaglinskas/Desktop/ROI_data.mat';
save(ofn,'roi_data')
%% Extract Voxel Data
if extract_voxel_data == 1;
n.subs = nsubs;
n.conds = nconds;
n.masks = length(masks.mat_files);
vx = {};
clc;
for r_ind = 1:n.masks
disp(sprintf('%d/%d',r_ind,n.masks));
R  = maroi(masks.mat_files{r_ind});
dt = {};
for i = 1:length({SPM.xY.VY.fname}); % All data
dt{i,1} = getdata(R,SPM.xY.VY(i).fname);
end % end subject task loop
dt = reshape(dt,n.conds,n.subs);
vx(r_ind,:,:) = dt;
end % ends roi loop
voxel_data.mat_files = masks.lbls;
voxel_data.dt = vx;
ofn = '/Users/aidasaglinskas/Desktop/voxel_data.mat';
save(ofn,'voxel_data')
disp('voxel data exported to');
disp(ofn)
end % ends if 
%% Fix SPM
fix_spm = 0;
if fix_spm;
old_str = '/Users/aidasaglinskas/Google Drive/Data/';
new_str = '/Users/aidasaglinskas/Google Drive/Aidas:Summaries & Analyses (WP 1.4)/Data_faces/';
SPM.xY.P = strrep(SPM.xY.P,old_str,new_str);
    b = strrep({SPM.xY.VY.fname},old_str,new_str)';
[SPM.xY.VY.fname] = deal(b{:});
save(spm_path,'SPM');
end