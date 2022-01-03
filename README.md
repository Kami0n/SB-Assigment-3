# Code for Assigment no. 3 for subject Image-based Biometry

This is my code for third assigment for subject Image-based Biometry.

## Goals of third assigment
In this assigment we will be learning about last two steps in biometric recognition pipeline:


## What did I do?
Tested LBP feature extraction without success. Can not get beter results (~4 %)  than Pix2Pix (~6 %) 

Trained my own CNN on default training set (Model 1).
Augmented data.
Trained my own CNN on augmented training set (Model 2).

Tested on test images that I extracted with my own extractor.

### Results:
Trained models:
| Model  | R1 | R5 |
| ------------- | ------------- | ------------- |
| Model 1 | 10.0 % | 24.0 % |
| Model 2 with augmented images | 26.8 % | 46.4 % |





### Notes:
Biometric identification involves comparing every record in the testing set against every other record in the testing set.

Identification rate is defined as the rate at which enrolled subjects are successfully identified as the correct individual,
where rank-k identification rate is the rate at which the correct individual is found within the top k matches.
Hence, the rank-1 identification rate is the rate at which the correct individual has the highest match score.

The cumulative match characteristic (CMC) plots identification rate by rank, for all ranks, and the areaunder-curve of the CMC, provides a metric by which to compare the accuracy achieved by CMC curves, where 100% AUC indicates perfect accuracy.