# Step 1: Create the Base OpenCV Application

![image alt text](../doc_support/step1_image_0.png)

# Introduction

This tutorial will show you the basics of what is needed to include and use OpenCV into an application. We will be working from the sample application that has already been created.  The sample is designed to be a minimal application that demonstrates how to use OpenCV functions to read image data and then display image data.  This tutorial will take a look at the OpenCV portions of the code and explain what is happening.  Then we will build and run the tutorial so we can see it in action.  In later tutorials, we will be adding the processing of the input image to this basic framework.

# The Basic OpenCV Application, Input and Output

Every application needs some way of getting data in and data out.  Let us now take a look at the code we will be using to do the input and output in our OpenCV application.  Then we can compile and run our program to see how it works using the base input and output settings.  

### Parsing Command Line Arguments

To make it easier to set everything from the input video file to which model and device is to be used, we will use command line arguments to our application.  To parse the command line arguments we make use of the "gflags" helper library that comes with the OpenVINO samples.  Here we will briefly go over the primary functions that were used, for reference the full source code for the gflags library may be found in the OpenVINO samples directory: /opt/intel/computer_vision_sdk/deployment_tools/inference_engine/samples/thirdparty/gflags

To make use of the gflags library and use the supplied functions and classes, we must include the main header file:

```cpp
#include <gflags/gflags.h>
```


We do this in the main header file "face_detection.hpp" where we will define all the command line arguments using these same steps:

#### Create the Argument

Create the argument using the macro "DEFINE_string".  Here we see how the “-i \<video filename\>” argument that we will use to specify the input video is defined:

```cpp
/// @brief message for images argument
static const char video_message[] = "Optional. Path to an video file. Default value is \"cam\" to work with camera.";

/// \brief Define parameter for set image file <br>
/// It is a required parameter
DEFINE_string(i, "cam", video_message);
```


In the above code:

* video_message[] is the argument’s help string

* DEFINE_string(i, "cam", video_message):

    * Specifies:

        * The string argument name as "i"

        * video_message as the help message

        * "cam" as the default value when not set

    * Creates the variable FLAGS_i to hold the string value for the "i" argument

#### Parse Arguments

In main.cpp’s main() function, ParseAndCheckCommandLine() is called to do command line argument parsing and checking for valid arguments. The actual argument parsing and setting variables is done by the call:
```cpp
gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
```
After returning, the FLAGS_i variable will be set with the "-i" command line arguments “\<video filename\>” value, or if not specified, the default value of “cam”.

This is how the "-i" argument is done, all other arguments are handled similarly using the other forms of the gflags macro according to data type needed as follows:

* DEFINE_uint32() for an unsigned 32-bit integer arguments

* DEFINE_string() for string arguments

* DEFINE_double() for double precision floating point arguments

* DEFINE_bool() for boolean arguments

### OpenCV Input to Output

1. Open up an Xterm window or use an existing window to get to a command shell prompt.

2. Change to the directory containing Tutorial Step 1:
```bash
cd tutorials/face_detection_tutorial/cd step_1
```

3. Open the files "main.cpp" and “face_detection.hpp” in the editor of your choice such as ‘gedit’, ‘gvim’, or ‘vim’.

4. First, we include a couple of header files that define some helpful utility classes we can use to simplify some of the day-to-day programming tasks, as well as some functions for making logging easier.

```cpp
#include <samples/common.hpp>
#include <samples/slog.hpp>
```


5. Then we are going to pull in the opencv.hpp file from the Intel OpenCV SDK.  This will give us access to the optimized OpenCV functions that Intel provides.

```cpp
#include <opencv2/opencv.hpp>
```


6. Now we jump down to the main function.  The first thing we need to do is to create an OpenCV video capture object that we will use to get image data.  Then we need to tell that object where to get the data.

```cpp
cv::VideoCapture cap;
if (!(FLAGS_i == "cam" ? cap.open(0) : cap.open(FLAGS_i))) {
   throw std::logic_error("Cannot open input file or camera: " + FLAGS_i);
}
```


7. FLAGS_i is a command line parameter that tells the program where the image data from is coming from.  It can be the path to an image file, the path to a video file, or cam.  If the parameter is cam, then the program will try to get input from the USB camera.

8. The next step is to create a place to store the image data and then read in the data.

```cpp
const size_t width  = (size_t) cap.get(CV_CAP_PROP_FRAME_WIDTH);
const size_t height = (size_t) cap.get(CV_CAP_PROP_FRAME_HEIGHT);
cv::Mat frame;
cap.read(frame);
```
9. Here, you can see that we are getting the width and height of the image file, or the camera resolution, and storing that for use later.  We also use cv::Mat to create an array that we can use to store the image data.  Then we read in the image data by calling "cap.read(frame)".

10. At this point, we have created OpenCV objects to read data, and store data, so we are ready to create our main loop to read in and then write out the image.  The loop will run forever until you press any key except for ‘s’ which will take a snapshot of the current output and save it as "snapshot.bmp", convenient for saving results to be used later. The main loop looks like:
    1. Run forever until we decide to break elsewhere inside the loop:
    ```cpp
    while (true) {
    ```
    2. We grab an available image from the source:
   
    ```cpp
    /** requesting new frame if any*/
    cap.grab();
    ```
    3. Check for key press to either snapshot (pressing ‘s’) or stop (any other key)

    ```cpp
    int keyPressed;
    if (-1 != (keyPressed = cv::waitKey(1)))
    {
   	 if ('s' == keyPressed) {
   		 // save screen to output file
   		 slog::info << "Saving snapshot of image" << slog::endl;
   		 cv::imwrite("snapshot.bmp", frame);
   	 } else {
   		 break;
   	 }
    }
    ```
    4. Show the output, we also wrap it with time functions to measure the time it took to do:

    ```cpp
    auto t0 = std::chrono::high_resolution_clock::now();
    if (!FLAGS_no_show) {
   	  cv::imshow("Detection results", frame);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    ocv_render_time = std::chrono::duration_cast<ms>(t1 - t0).count();
    ```


    6. Check for another image by retrieving it, if nothing to retrieve we wait for a key press if the command line "-no_wait" was not used.

    ```cpp
    // end of file, for single frame file, like image we just keep it displayed to let user check what was shown
    cv::Mat newFrame;
    if (!cap.retrieve(newFrame)) {
   	  if (!FLAGS_no_wait) {
   		   slog::info << "Press 's' key to save a snapshot, press any other key to exit" << slog::endl;
   		   while (cv::waitKey(0) == 's') {
   			 // save screen to output file
   			 slog::info << "Saving snapshot of image" << slog::endl;
   			 cv::imwrite("snapshot.bmp", frame);
   		   }
   	  }
   	  break;
    }
    frame = newFrame;  // shallow copy
    ```


    7. Let user know they can stop a multi-image source like video or camera

    ```cpp
    if (firstFrame) {
   	  slog::info << "Press 's' key to save a snapshot, press any other key to stop" << slog::endl;
    }

    firstFrame = false;
    ```


    8. End of loop, go back to while() to run again

    ```cpp
    }
    ```


# Building and Running

Now that we have looked at the code and understand how the program works, let us compile it and see it in action.

1. First, we need to configure the build environment when using OpenVINO by running the "setupvars.sh" script.

```bash
source  /opt/intel/computer_vision_sdk/bin/setupvars.sh
```


2. Now we need to create a directory to build the tutorial in and change to it.

```bash
mkdir build
cd build
```


3. The last thing we need to do before compiling is to configure the build settings and build the executable.  We do this by running CMake to setup the build target and file locations.  Then we run Make to build the executable:

```bash
cmake -DCMAKE_BUILD_TYPE=Release ../
make
```


4. You should now have the executable "face_detection_tutorial" file in the “./intel64/Release/” directory.  We will be using that executable to run our base OpenCV application.

5. Now, it is time to run our application.  We will run it using each type of input (image file, video file, camera) so you will know what to expect.  We have included commands that will have the application load images or videos that come with OpenVINO and this tutorial, but you can also use your own images.  If the application cannot find an image, or if you have not connected the USB camera to the UP Squared board, it will print an error message and return to the command prompt.  If that happens, check the path to the image or video file, to make sure it is correct and try again.

6. First, let us use our application to view a single image file.  We do this by using a "-i" parameter followed by the name of an image file.

```bash
./intel64/Release/face_detection_tutorial -i ../../data/face.jpg
```


7. You should now see a new window with an image.  You should also see a "Press 's' key to save a snapshot, press any other key to exit" prompt in the console window.  The application will now wait for you to press a key with the image window active.

    1. Note: Pressing a key in the console window will not do anything because the image window is detecting key presses.  Use Ctrl+C to exit.

8. Next, let us see how our application handles a video file:

```bash
./intel64/Release/face_detection_tutorial -i /opt/intel/computer_vision_sdk/openvx/samples/samples/face_detection/face.mp4
```


9. You will now see a window appear and play the video.  After the video has finished playing, the window will continue to display the final frame of the video, waiting for you to press a key with the image window active.

10. Finally, we can use the application to view live video from the USB camera connected to the UP Squared board.  The camera is the default source, so we do this by running the application without using any parameters.
    ```bash
    ./intel64/Release/face_detection_tutorial
    ```
    2. Or we can still specify the camera using "cam":
    ```bash
    ./intel64/Release/face_detection_tutorial -i cam
    ```
11. You will now see the output window appear displaying live input from the USB camera.  When you are ready to exit the application, make sure the output window is active and press a key.

# Conclusion

Now we have seen what it takes to create a basic application that uses OpenCV to read and display image data.  We have also seen how our application works with each type of image input it accepts including still images, video files, and live video from the USB camera.  We will be using the basic framework from this step of the tutorial as we move forward building up the application step-by-step.  Next, in Tutorial Step 2 we will be adding the ability to process images and actually detect faces.

# Navigation
[Face Detection Tutorial](../Readme.md)

[Face Detection Tutorial Step 2](../step_2/Readme.md)


