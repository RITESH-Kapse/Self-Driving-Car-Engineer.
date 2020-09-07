For Complete explanation , please refer :
https://onedrive.live.com/edit.aspx?resid=371BFBA6D73CBD9E!345&ithint=file%2cdocx&authkey=!AmgEsu7bihzJ3YQ



Project 1 - Finding Lane Lines on the Road 

  
The goals / steps of this project are the following: 

      Make a pipeline that finds lane lines on the road 
      Reflect on your work in a written report 
      Get more idea about lane detection fundamentals 
      First step towards documentations 
      Identify potential shortcomings 
       Possible improvements 


My pipeline consisted of below steps : 

      Conversion of images to gray scale 
      Finding region of interest 
      Gaussian smoothing to remove the noise and blur the image . 
      Canny Edge detection over gaussian image 
      Apply the Hough transform on canny image. 
      Applying these results on actual image as weighted image. 
      Averaging the lane lines 
      Applying results on video( which is nothing but series of Images) 

How to run the project :

      Simply open "P1 Lane Detection Final Code.ipynb" file and run the cells inside jupyter notebook. Results will get saved to "test_videos_output" folder.
