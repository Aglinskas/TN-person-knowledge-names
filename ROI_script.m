names = {'PREC-M'
'AG-L'
'AG-R'
'vmPFC-M'
'dmPFC-M'
'IFG-L'
'OFC-L'
'ATL-L'}

coords = [0	-49	26
-45	-61	26
51	-67	35
0	50	-19
6	62	20
-51	26	11
-39	35	-13
-60	-10	-19]

ofn = '/Users/aidasaglinskas/Google Drive/Aidas/Data_words/ROIs_8/';
sph_radius = 8;
space_fn = '/Users/aidasaglinskas/Google Drive/Aidas/Data_words/Group_Analysis_subconst/beta_0010.nii';
blobs_dir = '/Users/aidasaglinskas/Google Drive/Aidas/Data_words/ROIs_8/Blobs/';
%% Make ROIs
func_makeROIsFromCoords_names(coords,names,ofn,sph_radius,space_fn,blobs_dir)
%% Extract ROI data
roi_dir = ofn;
spm_dir = '/Users/aidasaglinskas/Google Drive/Aidas/Data_words/Group_Analysis_subconst/';
func_extract_data_from_ROIs(roi_dir,spm_dir)