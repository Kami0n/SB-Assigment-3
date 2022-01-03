# Code for Assigment no. 3 for subject Image-based Biometry

This is my code for third assigment for subject Image-based Biometry.

## Goals of third assigment
In this assigment we will be learning about last two steps in biometric recognition pipeline:


## What did I do?
1. Tested LBP feature extraction without success. Can not get beter results (~4 %)  than Pix2Pix (~6 %) 
<br />

2. Trained my own CNN on default training set (Model 1).
3. Augmented data.
4. Trained my own CNN on augmented training set (Model 2).
<br />

5. Tested on test images that I extracted with my own extractor.

### Results:
Trained models:
| Model  | Rank-1 | Rank-5 |
| ------------- | ------------- | ------------- |
| Pix2Pix | 6,41 % | / |
| LBP | 4,27 % | / |
| CNN model 1 | 10,0 % | 24,0 % |
| CNN model 2 with augmented images | 26,8 % | 46,4 % |


![alt text](https://github.com/Kami0n/SB-Assigment-3/blob/main/results/model_1_CMC.png?raw=true)
![alt text](https://github.com/Kami0n/SB-Assigment-3/blob/main/results/model_2_augmented_CMC.png?raw=true)




<br /><br /><br /><br /><br /><br />

### Notes:
Biometric identification involves comparing every record in the testing set against every other record in the testing set.

Identification rate is defined as the rate at which enrolled subjects are successfully identified as the correct individual,
where rank-k identification rate is the rate at which the correct individual is found within the top k matches.
Hence, the rank-1 identification rate is the rate at which the correct individual has the highest match score.

The cumulative match characteristic (CMC) plots identification rate by rank, for all ranks, and the areaunder-curve of the CMC, provides a metric by which to compare the accuracy achieved by CMC curves, where 100% AUC indicates perfect accuracy.