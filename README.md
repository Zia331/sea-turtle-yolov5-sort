# Sea Turtles tracking by YOLOv5 and SORT

This project has 3 major steps to achieve the purpose of getting the total amount of sea turtles in a drone video.

1. Detecting the sea turtles using YOLOv5.
2. Tracking the sea turtles detected using SORT algorithm.
3. Calculate the threshold for the video to minimize the counting error.

After these 3 steps, we get the number of sea turtles detected in a given drone video.

## Cites
### SORT
```
@inproceedings{Bewley2016_sort,
  author={Bewley, Alex and Ge, Zongyuan and Ott, Lionel and Ramos, Fabio and Upcroft, Ben},
  booktitle={2016 IEEE International Conference on Image Processing (ICIP)},
  title={Simple online and realtime tracking},
  year={2016},
  pages={3464-3468},
  keywords={Benchmark testing;Complexity theory;Detectors;Kalman filters;Target tracking;Visualization;Computer Vision;Data Association;Detection;Multiple Object Tracking},
  doi={10.1109/ICIP.2016.7533003}
}
```
### YOLOv5
```
@software{glenn_jocher_2020_4154370,
  author = {Glenn Jocher},
  title = {{ultralytics/yolov5: v6.0 - YOLOv5n 'Nano' models, Roboflow integration, TensorFlow export, OpenCV DNN support}},
  month = oct,
  year = 2021,
  publisher = {Zenodo},
  version = {v6.0},
  doi = {10.5281/zenodo.5563715},
  url = {https://doi.org/10.5281/zenodo.5563715},
  howpublished = {\url{https://github.com/ultralytics/yolov5}}
}
```
