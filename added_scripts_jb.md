# Scripts and notes Jan Bijster
*last update 05-03-2018*

## Plotting results
### extract_body_parts_plot.py
Script to plot images in a folder, overlayed with lines indicating the body
parts found.

## Comparing poses
### normalize_parts.py
Contains a function for normalizing a found pose: center and scale the
coordinates so that two poses can be compared more easily.
### extract_body_parts_store.py
Script to analyze all images in a folder (inc subfolders) and store the
found poses in a json file.
### compare_pose.py
Contains a function to compare a pose with a large number of preanalyzed poses.
### test_compare_pose.py
Script to test the compare_pose function
### results
![Comparing two poses result](/results/pose_match.jpg)
I encountered the following questions/problems/challenges:
* What to do with missing parts (now the image is discarded)
* How to match the pose comparison to our intuition (what we consider similarity in pose)

## PCA analysis
### coco_to_openpose.py
Contains a function to convert pose annotations as found in the COCO dataset to
a format like returned by the tf-pose-estimation algorithm.
### read_annotations.py
Script to read and convert all the annotations in an annotated person-keypoints
dataset from the COCO datasets.
(I have not included the datasets in this repository, because they are quite
large. The result of this script is included, so you can skip it.)
### eigenpose.py
Script to perform PCA on processed pose data and plot the found component
vectors.
### results
Using 8475 full poses from the COCO 2017 person-keypoints training set:
![PCA analysis result](/results/eigenposes.jpg)
