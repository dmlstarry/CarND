# Behavioral Cloning Project

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I'm using the NVIDIA model, just adding the parameters of the full connection layer.

#### 2. Attempts to reduce overfitting in the model

I can finish the track at the fastest speed without using regularization technology.Because of the time factor, this one will be tested later.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually .

#### 4. Appropriate training data

I used the data provided by Udacity, and of course I recorded a few laps of data myself, trying to keep the car in the middle of the driveway.To ensure that there are more angles of data, when cross the bend, pounding on the direction key

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

At first I used only the middle camera data, but always fell in a bend.
It then used data from the left and right cameras and doubled the amount of data on the flip.

To confirm that the generator has no problem, use reverse data training and then use positive data training.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture  consisted of a convolution neural network with the following layers and layer sizes:

![](./Image/summary.PNG)

#### 3. Here are a few tips
* Don't cut too much
* Remember to convert the color field to RGB
* Sometimes there are no local images, and actively skip them
* You can train your data on a particular bend

 

