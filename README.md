# eye_mouse_movement
This project uses two sources of data: Four youtube videos and two CSV files.

The four youtube videos are:
- [Condition 1: No gaze, no mouse](https://www.youtube.com/watch?v=0G3Q2WmW-fQ)
- [Condition 2: No gaze, mouse](https://www.youtube.com/watch?v=bXXcUhmBFiM)
- [Condition 3: Gaze, no mouse](https://www.youtube.com/watch?v=zvIjUT_wWPQ)
- [Condition 4: Gaze and mouse](https://www.youtube.com/watch?v=iqU_BxtKP80)

The two CSV files:
- One of 124 responses in total to each video (31 for each video) answering questions such as clarity on following what the teacher was referring to and how much mental effort watching the video took for each student.
- The other is a text file of all the variables in the CSV file with their descriptions. 

## Files Included
### 1_applyGrid.py
Uses one of the three video files downloaded from youtube (2-4) to apply a grid to each frame.
Within this program, need to uncomment/comment as needed according to if you want to track the mouse or the gaze. 
The output will be coordinates of either the mouse or the gaze as well as the timestamp in ms, printed to the console.

### 2_fillSteps.py
This prorgam takes in a file (printed results from 1_applyGrid.py), and fills in gaps in the original trajectory of coordinates. 
The output is complete and continuous trajectory data with the corrdinates, the action/direction that the mouse/gaze is moving in, the object number if the mouse/gaze is on one, and the timestamp in ms. 

### grid_world.py
Customized gym enviroment class using gym library.
Requires more testing and debugging.
More details on how to use can be found [here](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/).

### policy.py
Main RL model details are located in this file which depends on utilization of the above grid_world.py enviroment. 
Closely follows [this tutorial](https://medium.com/@sthanikamsanthosh1994/imitation-learning-behavioral-cloning-using-pytorch-d5013404a9e5).
Incomplete, requires futher testing and debugging. 
