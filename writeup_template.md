## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./writeup_data/samples.png "Samples"
[image2]: ./writeup_data/hog.png "Hog Sample"
[image3]: ./writeup_data/search.png "Search on Test Image"

[image4]: ./writeup_data/pipeline1.png "Pipeline 1"
[image5]: ./writeup_data/pipeline2.png "Pipeline 1"
[image6]: ./writeup_data/pipeline3.png "Pipeline 1"
[image7]: ./writeup_data/pipeline4.png "Pipeline 1"
[image8]: ./writeup_data/pipeline5.png "Pipeline 1"

[image9]: ./writeup_data/heat1.png "Heat 1"
[image10]: ./writeup_data/heat2.png "Heat 1"
[image11]: ./writeup_data/heat3.png "Heat 1"
[image12]: ./writeup_data/heat4.png "Heat 1"
[image13]: ./writeup_data/heat5.png "Heat 1"

[image14]: ./writeup_data/label1.png "Label 1"
[image15]: ./writeup_data/label2.png "Label 1"
[image16]: ./writeup_data/label3.png "Label 1"
[image17]: ./writeup_data/label4.png "Label 1"
[image18]: ./writeup_data/label5.png "Label 1"

[image19]: ./writeup_data/final1.png "Final 1"
[image20]: ./writeup_data/final2.png "Final 1"
[image21]: ./writeup_data/final3.png "Final 1"

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the fifth code cell of the IPython notebook.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is examples of a few of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `RGB` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(8, 8)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters. For each combination that I considered, I ran feature extraction (8th cell of notebook) on all images and then trained the SVM classifier. This is table of all the considered parameters:

Label | Colour Space| Orientaion | Pixels Per Cell | Cell Per Block | HOG Channel | Accuracy  | Extract Time
------| ----------- |----------  | ----------------| ---------------| ----------- | ----------| -------------
1     |    RGB      |      9     |         8       |       2        |     all     |   0.9758  |   215.23
2     |    HSV      |      9     |         8       |       2        |      1      |   0.9186  |    96.75
3     |    HSV      |      9     |         8       |       2        |      2      |   0.9609  |    85.37
4     |    LUV      |      9     |         8       |       2        |      0      |   0.9623  |    79.88
5     |    LUV      |      9     |         8       |       2        |      1      |   0.9406  |    87.34
6     |    HLS      |      9     |         8       |       2        |      0      |   0.9333  |    78.61
7     |    HLS      |      9     |         8       |       2        |      1      |   0.9623  |    73.54
8     |    YUV      |      9     |         8       |       2        |      0      |   0.9651  |    76.66
9     |    YUV      |      9     |         8       |       2        |     all     |   0.9820  |   218.69
10    |    YUV      |      11    |         8       |       2        |     all     |   0.9828  |   233.13
11    |    YUV      |      11    |         16      |       2        |     all     |   0.9814  |    95.61
12    |    YUV      |      5     |         8       |       2        |     all     |   0.9803  |   113.67

Based on these results, I chose the parameters with label 10 which had the highest accuracy.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM on 9th cell of the notebook. I used the default parameters of the SVM and the HOG parameters specified above. I was able to achieve accuracy of 98.28%.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I adapted the method `find_cars` from the lessons. In the lessons the instructor suggested we sample a region then run the HOG feature extraction on the region of the image. However, instead I ran HOG feature extraction on the whole image and then performed the sliding window search which turned out to be less time consuming.

This what the sliding window search looks like on a sample image:

![alt text][image3]

It found the two cars without any falsepositive. However, cars on a video can be of different sizes and position on the screen. So ran sliding window search several times with different window height, subsample size, and scale factor.

These are final pipeline runs the sliding window search five times with the following parameters:

| Label | yStart | yStop | scale |
|-------|--------|-------|-------|
|   1   |   400  |  464  |  1.0  |
|   2   |   416  |  480  |  1.0  |
|   3   |   400  |  496  |  1.5  |
|   4   |   432  |  528  |  1.5  |
|   5   |   400  |  528  |  2.0  |

I decided on these values byt thinking about different sizes of car a driver sees while driving.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on only using all YUV channes HOG features and din't use spatially binned color and histograms of color in the feature vector, which provided a nice result. Here are some example images:

![image4]
![image5]
![image6]
![image7]
![image8]

By picking the HOG parameters based on the accuracy of the crassifier turned out to be a good approach and didn't need to optimize the classifier any further.

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./writeup_data/videoOutput.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are three frames and their corresponding heatmaps:

![image9]
![image10]
![image11]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![image14]
![image15]
![image16]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![image19]
![image20]
![image21]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The main issues that I had were with the parameter tuning for HOG. I had to make a choice between speed of the pipeline compare with accuracy. I got the classifier to 98% which works well but to improve it further the pipeline computation speed might get slower. 

The pipeline will most likely fail if the on coming traffic is fully visible and will classify those cars as well (it might not be desired to classify on coming cars). Also, just like lane detection it is very dependant on the lighting condition. It can also fail if a new type of car is seen that looks very different than the training data.

To make it robust the classifier can be fine tuned by cosidering different parameters that were mentioned on the lessons. Also have a high overlap in the window search will result in better performace at the cost of speed.  

