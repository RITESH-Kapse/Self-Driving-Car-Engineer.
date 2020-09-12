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

Referred Links/Videos :
      https://www.youtube.com/watch?v=hnXkCiM2RSg&feature=youtu.be
      https://www.youtube.com/watch?v=LECg-Gv5xjo&list=PLJeClhKBNwzhHQA9moIPMPB6Ly5PNjbtm&index=46
      https://medium.com/@naokishibuya/finding-lane-lines-on-the-road-30cf016a1165
      https://github.com/naokishibuya/car-finding-lane-lines/blob/master/Finding%20Lane%20Lines%20on%20the%20Road.ipynb
      https://stackoverflow.com/questions/21324950/how-can-i-select-the-best-set-of-parameters-in-the-canny-edge-detection-algorith
      
