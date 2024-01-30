----------------------------------------------------------------
SRC_GROUPING
----------------------------------------------------------------

This module is designed for the specific task of grouping segmented voxels 
into nodules for lung nodule detection in CT scans. It takes as input the 
voxel segmentations flagged by the segmentation model (denoted as SRC_SEGMENTATION) 
and groups these segmented voxels into candidate nodules. This process is achieved 
by applying a set threshold to the model predictions and subsequently grouping connected 
regions of flagged voxels. The output of this module is a list of candidate nodules, 
each annotated with coordinates to their respective centers. These coordinates are 
intended for use in a subsequent classification model, which distinguishes between
 nodules and non-nodules.