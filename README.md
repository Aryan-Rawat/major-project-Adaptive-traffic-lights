# major-project-Adaptive-traffic-lights
Using object detection for Adaptive traffic lights which can detect number of vehicles and release densely packed lanes.

# Repository for Details

## Points to ponder :)

- [x] understand (YOLOv5 algorithm)
- [X] Use SSD (Single Shot Detection) model for comparision
- [x] use formula for calculating the traffic density using the information extracted by our deep learning model
- [x] Write an algorithm for managing traffic light efficiently using multiple threads for real time results
- [x] Write scripts for syncing model results, traffic light management algorith and GUI 

## Instructions for running the code (On linux/unix system)

```
	pip install -r requirements.txt (First time only, for installing all the required dependencies)
	./run.sh
```


## An attempt to reduce traffic congestion using deep learning for object detection and traffic light time management.

Once the live feed is fed to the deep neural network it returns the location of traffic elements, their count and their type (cars, buses, etc.). This information is used for the calculation of traffic density using a formula. Once we have the traffic density, the model releases traffic from each lane according to density.



### YOLOv5 results

<img
	src=./east.jpg
	align="center"
/>

<img
	src=./east_bbox.jpg
	align="center"
/>

### Work Flow Model

<img
	src=./workflow.JPG
	align="center"
/>
