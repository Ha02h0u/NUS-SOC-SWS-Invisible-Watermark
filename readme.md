# NUS-SOC-SWS-Invisible-Watermark
## program

Requirement: 64-bit Windows System

Necessary: Unzip ffmpeg.zip to get ffmpeg.exe in the same folder. (It's because Github doesn't allow files more than 100MB)

Note: It's recommended to put your original video in program folder.

## source
watermark.cpp - C++ Code, charge of frame watermark process

Requirement: [OpenCV 2.4.13.6 environment](https://github.com/opencv/opencv/archive/2.4.13.6.zip)

Video_Watermark.e - [Easy Programming Language](http://www.dywt.com.cn/) Code, charge of Windows program part

Requirement: ffmpeg environment

##attack
mani.py - (stands for 'manipulate') Python code, charge of various attack to the video

Requirement: cv2 numpy 
	You may simply install them by `pip install cv2 numpy`

## how to attack

```shell
Usage: python3 mani.py [options] [source_video] [dest_video]
    options: cut_height/cut_width/resize/bright/rotation/shelter/salt_pepper
    example: python3 mani.py cut_height ./watermark.mp4 ./attacked.mp4
```

You will also see this information when you use it in wrong format. 
