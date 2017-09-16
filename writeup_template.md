# **Finding Lane Lines on the Road** 

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./test_images/solidWhiteRight.jpg_LINE.jpg

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

First, I converted the images to grayscale, then I use cv2.GaussianBlur to smooth out edges.
cv2.Canny to find edges.cv2.HoughLinesP to find the lane line in the part that needs to be detected.

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by calculate average slope and intercept for the left and right lanes of each image,then converted to coordinates in order to draw lines.

The final output line image is combined with the original image.

If you'd like to include images to show how the pipeline works, here is how to include an image: 

![alt text][image1]


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when the corners of the corners are detected, the deviation is very large

Another shortcoming could be the length of the dotted line is different,resulting in the painting line is not smooth.


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to find a solution that can also be applied to the curve.

Another potential improvement could be to adjust the parameters to make the program more stable.
