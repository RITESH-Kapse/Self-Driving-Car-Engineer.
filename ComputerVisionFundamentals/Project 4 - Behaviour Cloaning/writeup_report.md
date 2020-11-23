

## Behavioural Cloning - Self Driving Car

Self driving car is always a facinating world where I always dreams of getting better each day. There are number of things to check before achieving that dream of becoming a self driving car engineer. So here we are moving one step closer with the help of Udacity's Simulator which helps to let us train the brain of car using Deep learning and CNN algorithms.

Being newbie to this field , I decided to follow the well defined and well known NVDIA architecture to implement the algorithm by little manipulations.

Also most of the code references are used as explanined inside the classroom step by step guide to approach this project.

IMP NOTE : When I tried started , I have installed CUDA and Cudnn in my machine where tensorflow and keras version were latest . But it gave multiple errors like model.fit() method was not getting accessed. So finally, i decided to follow the udacity GPU access provided.

##### Below are the files submitted

model.py - This is used to create and train the model

drive.py - This is as it is provided by udacity which consists of PID controllers to drive the car.

model.h5 - This saves the model and used to help simulator to drive the car

writeup_report - This  consist of detailed analysis of project and learnings.

video.mp4 - Video recording of vehicle driving autonomously for one round around the track.

Note : I analysed the data provided by udacity and found that has more than 7 laps and  I can proceeed with same for this project.

##### Code Used during execution

To generate model.h5 file and run the basic architecture on dataset use :

```This
python model.py
```

To run the simulator on autonomous mode after training and implementation of model on test data use below command :

```This
python drive.py model.h5 
```

To perform video recording of autonomous mode  :

```
python drive.py model.h5 run1

#Here run1 is the folder name where we are storing the results 
#after car runs autonomously
```

Now based on the images found in run1 directory , if you want to create the video :

```
python video.py run1
```

Note : The name of the video will be the name of the directory followed by '.mp4', so, in this case the video will be run1.mp4. But we want output video as video.mp4 so we can rename while running command of after generation of video.

To specify frames per second , use :

```
python video.py run1 --fps 48
```

##### Model Overview

Reference : https://developer.nvidia.com/blog/deep-learning-self-driving-cars/

This algorithm is tested on Defense Advanced Research Projects Agency (DARPA)  Autonomous Vehicle (DAVE) by NVDIA.

For data collection, three cameras are mounted on windshield of car, and video is captured by noticing the sterring angles provided by human at particular stage.

Below is the high level view of data collection system :

<div><div>
<img src = "Dave-2 System NVDIA.png" alt = "architecture" >
</div>

</div>

Now only considering the center camera or inputs from humans are not suffiecient to train the network. Hence we need to consider all three cameras and random shift . Also need to flip an images since the training track mostly has left turns, so if in case vehicle is off road, it should come back to center and complete the lane. This way from one image we can generate 6 different input images for the network to train .

Below is the way we are training our network :

<div><iv>
<img src = "training the network.png" alt = "architecture" >
</div>

##### Dataset

I have used dataset provided by udacity to train my network

Used OpenCv to read the images, which reads image in BGR format but to process the images by our simulator , we need it in RGB format . Hence converting.

As suggested by network architecute , we are using three images as center  , left and right to detect the sterring angle . So we are adding certain amount of correction factor .  It is trial and error method where I am trying with 0.15

Sample Image from dataset provided by udacity  :

<div><div>
<img src = "center_2016_12_01_13_30_48_287.jpg" alt = "dataset" >
</div>

Now clearly we can see that the dome part of car is visible and also sky and trees are visible , so we need to preprocess the images

As suggested by udacity tutorials, going to cut down upper part (70 pixels) and lower dome part(25 pixels) so that we only can see the road surface part.

Also since most of the training laps has left turns, its difficult for architecture to learn when vehicle is off track and to train the network we are using image augumentation. Where we will flip the image (-1) and get its sterring angle . By this way we are getting more dataset to train the algorithm.

By using sklearn preprocessing library , dataset is splitted into training and test dataset with training data to be 80 percent and validation data be 20 percent.

As suggested by udacity , tried to use the generator function which will yield the images and respective angles at batch size of 32 ( which is standard pracitce.)

The driving_log.csv file is provided by simulator with dataset (IMG folder), which we are going to use during implementations.

## Final Model Architecture

Model is completely based upon NVDIA's DAVE-2 car with certain modifications mentioned below.

To control the overfitting problem, used dropout layers added after fully connected layers.

Used ELU activation function instead of RELU for hidden layers

To keep the steering angle prediction output in range of +1 to -1 , tanh activation used at output neuron. Since it is regression problem , we are taking single output layer.

Mean square error is used with adam optimizer to compile the model.

Also input images size taken by model are (60,266,3) and available images size are (160,320,3). So reshape of input images are performed.

NVDIA's DAVE-2 car architecture model image :

<div><div>
<img src = "NVDIA DAVE-2.png" alt = "dataset" >
</div>

Type of convolution and fullly conncted layers are as it is used as per NVDIA's Architecture.

Parameters tunning performed as follows :

epochs - 2

Optimizer - Adam

Learning Rate- 0.001

Validation Data split- 0.20

Generator batch size -32 ( This is standard)

Correction factor- 0.15

Loss Function Used- MSE(Mean Squared Error)

##### Layers defined :

1. At first , we performed image normalization and the normalizer is written once and not adjusted during training process.

2. Convolutional layers are used to perform feature extraction and so we have used strided convolutions with filter depth of 24 , filter size of (5,5) and  stride of 2. Also used ELU activation layer.

3. Second convolution layers with strided convolutions has filter depth of 36 , size of (5,5) and stride of 2 , followed by ELU activation function.

4. Third convolution layers with strided convolutions has filter depth of 48 , size of (5,5) and stride of 2 , followed by ELU activation function.

5. Now we are using TWO 3*3 convolution layers, where filter depth is 64 , size is (3,3) and stride of 1 followed by ELU activation layer.

6. Now we need to flatten the output before passing to fully connected layers.

7. We build three fully connected layers , after each layer we are taking care of dropouts to make our model safe from overfitting.

8. First fully connected layer has output 100 and then introduce dropout with rate of 0.5

9. Now second fully connected layer with 50 outputs and introduce dropout with rate of 0.5 again.

10. Then third connected layer with 10 outputs and dropout of 0.5.

11. Finally since it is regression problem , we have final layer with one output.



Model Summary :

<div><div>
<img src = "modelsummary1.png" alt = "dataset" >
</div>



<div><div>
<img src = "modelsummary2.png" alt = "dataset" >
</div>

## Output Video- Autonomous mode

<video width="320" height="240" controls>
  <source src="video.mp4" type="video/mp4">
</video>
 

## Final Thoughts

This was the one of toughest project I have interacted being newbie to this field after advance lane detections.

I understand that , by training the machine, we dont need to rely on lane detections particularly to drive the vehicles.  But while this learning process , I came across below findings :

1. When i installed CUDA and Cdnn on local machine and tried to build model.py file , I found that most of the errors are encounters due to compatibility issues like model.fit() method was not getting executed. So I tried seeking help in forums and noticed that i need to downgrade the completed installations.

2. Finally I tried to get my own data and train the network since it was exciting too. But my local machine configurations continued to give errors and finally I ended up using udacity GPU platform.

3. There as well I tried to add my own dataset and perform the operations, but the simulator available on GPU platforms lags with minimal delay which causes vehicle to go offroad.

4. So, finally  I decided to complate the project using given dataset by udacity.

5. Lesson Learned: Meanwhile it takes more than three weeks to do above process since I am working associate , I tried to refer certain github and medium resources and ended up getting plagarism warnings which could turn down my degree. 

**Thank you for the opportunity !**
