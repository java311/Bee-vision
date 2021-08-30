31/08/2020 
These are the programs that gave me the best results. 
- Use only charuco board for calibration.
- keep an eye on the resolution (all of them have to have the same resolution)
- keep an eye con the charuco board you are using
- the charuco points are computed by me (in mm)
- To use stereo stereoCharucoCalib.py may be enough. 
- the camera rig was aligned to the right(from front) RS infrared camera. with the gray stand. 
- RMS value is gold
- check the order of the matrices it is important  M1 D1 etc. 
- pay attention to the distortion array format (this can create problems) 


FOR the UV FILTER
- Rt matrix and T vector need to be scaled Rt*1.3 
  (because the scaling facor of the focus lenght... ??) 
  and T/1000 (because the mm ??) I do not know.
- the calibration matrix of the RS infrared camera comes directly from RS API




TODO
- The cameras need to be aligned in X for them to calibrated with StereoCalib
- print the stupid green lines in the rectification pictures. 
- check on how to extract  