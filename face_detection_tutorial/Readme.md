# Introduction

The purpose of this tutorial is to examine a sample application that was created using the OpenVINO(TM) toolkit and UP Squared hardware included in the UP Squared AI Vision Development Kit.  The application is able to run inference models on the CPU, GPU and VPU devices to process images.  The models can be used to process video from the USB camera, an existing video file, or still image files.  To do that, we will download the latest Face Detection Tutorial from GitHub and then walk through the sample code for each step before compiling and running it on the UP Squared hardware.

This tutorial will start from a base application that can read in image data and output the image to a window.  From there, each step adds deep learning models that will process the image data and make inferences.  In the final step, the complete application will be able to detect a face, reports age and gender for the face, and draws a 3D axis representing the head pose for the face.  Before that, some key concepts related to and for using the OpenVINO toolkit will be first introduced and later seen along the way within the steps.  

# Getting Started

## Prerequisites

The UP Squared AI Vision Development Kit comes ready to go with all the hardware needed for this tutorial and is fully preconfigured with all software tools, libraries, drivers, etc. needed.   A summary of what is used:

* Hardware

    * From the kit:

        * UP Squared Board

        * AI Core mPCIe board (installed), this is what is being referred to as the "Myriad"

        * USB Camera

        * Power supply

    * User supplied:

        * USB keyboard and mouse

        * HDMI or DisplayPort cable and monitor

        * Ethernet cable

* Software (pre-installed in the kit)

    * OpenVINO toolkit

        * Inference Engine with plugins support for CPU, GPU, and Myriad

        * Optimized OpenCV and OpenVX libraries

        * Samples and common helper libraries

By now you should have completed the setup and getting starting guide for the kit, however before continuing, please ensure that:

* You have followed all the steps in the getting starting guide for your UP Squared AI Vision Development Kit.  This tutorial assumes that you have already setup and run the supplied test samples to test that your kit is fully functional including:

    * The UP Squared board is booted and running 

    * The USB camera is connected and operating correctly

* Your UP Squared board is connected to a network and has Internet access.  To download all the files for this tutorial, the UP Squared board will need to access GitHub on the Internet. 

## Downloading the Tutorial from the Git Repository

The first thing we need to do is create a place for the Face Detection tutorial and then download it.  To do this, we will create a directory called "tutorials" and use it to store the files that we download from the “cvs-sdk-tutorial” GitHub repository.  There are two options to download this tutorial: 1) Download as part of the entire repository using “git clone”, or 2) Use “svn export” to download just this tutorial (smaller)

### Using Git Clone to Clone the Entire Repository

1. Bring up a command shell prompt by opening an Xterm window or selecting an Xterm window that is already open.

2. Create a "tutorials" directory where we can download the Face Detection tutorial and then change to it:

```Bash
mkdir tutorials
cd tutorials
```


3. Clone the repository:

```Bash
git clone https://github.com/intel-iot-devkit/cv-sdk-tutorials.git
```


4. Change to the face detection tutorial folder:

```Bash
cd cv-sdk-tutorials/face_detection_tutorial
```


### Using SVN Export to Download Only This Tutorial

1. Bring up a command shell prompt by opening an Xterm window or selecting an Xterm window that is already open.

2. Create a "tutorials" directory where we can download the Face Detection tutorial and then change to it:

```Bash
mkdir tutorials
cd tutorials
```


3. Download the subdirectory for just this tutorial from the repository:

```Bash
svn export https://github.com/intel-iot-devkit/cv-sdk-tutorials.git/trunk/face_detection_tutorial
```


4. Change to the face detection tutorial folder:

```Bash
cd face_detection_tutorial
```


Now that we have all the files for the Face Detection Tutorial, we can take some time to look through them to see what each part of the tutorial will demonstrate.

### Tutorial FIles

In the "face_detection_tutorial" directory you will see:

* cmake\ - Common CMake files 

* data\ - Image, video, model, etc. data files used with this tutorial

* doc_support\ - Supporting documentation files including images, etc.

* scripts\ - Common helper scripts

* step_1\ - Tutorial Step 1: All files including Readme.md documentation

* step_2\ - Tutorial Step 1: All files including Readme.md documentation

* step_3\ - Tutorial Step 1: All files including Readme.md documentation

* step_4\ - Tutorial Step 1: All files including Readme.md documentation

* Readme.md - The top level of this tutorial (this page)

## OpenVINO Toolkit Overview and Terminology 

Let us begin with a brief overview of OpenVINO toolkit and what this tutorial will be covering.  The Open Visual Inference & Neural Network Optimization (OpenVINO) toolkit enables the quick deployment of convolutional neural networks (CNN) for heterogeneous execution on Intel hardware while maximizing performance. This is done using the Intel Deep Learning Deployment Toolkit included within OpenVINO with its main components shown below.

![image alt text](./doc_support/step0_image_0.png)

The basic flow is:

1. Use a tool, such as Caffe, to create and train a CNN inference model

2. Run the created model through Model Optimizer to generate a set of Intermediate Representation (IR) files (.bin and .xml) that is optimized for use with the Inference Engine

3. The User Application then loads and runs models onto devices using the Inference Engine and the IR files  

This tutorial will focus on the last step, the User Application and using the Inference Engine to run models on CPU, GPU, and Myriad.  

### Using the Inference Engine

Below is a more detailed view of the User Application and Inference Engine:

![image alt text](./doc_support/step0_image_1.png)

The Inference Engine includes a plugin library for each supported device that has been optimized for the Intel hardware device CPU, GPU, and Myriad.  From here, we will use the terms "device" and “plugin” with the assumption that one infers the other (e.g. CPU device infers the CPU plugin and vice versa).  As part of loading the model, the User Application tells the Inference Engine which device to target which in turn loads the associated plugin library to later run on the associated device. The Inference Engine uses “blobs” for all data exchanges, basically arrays in memory arranged according the input and output data of the model.

#### Inference Engine API Integration Flow

Using the Inference API follows the basic steps:

1. Load plugin

    1. Load the plugin for a specified device

2. Read model IR

    2. Read in IR files

3. Configure input and output

    3. Probe model for input and output information

    4. Optionally configure the precision and memory layout of inputs and outputs

4. Load model

    5. Load the model into the plugin

5. Create inference request

    6. Have plugin create a request object that holds input and output blobs

6. Prepare input

    7. Get an input blob to hold input data

    8. Transfer data from source into input blob

7. Infer

    9. Request plugin to perform inference and wait for results

8. Process output

    10. Get output blobs and process results

In tutorial Steps 2, 3, and 4 we will walkthrough the code that specifically integrates each of the models used in our application.  More details can also be found in the "Integrating Inference Engine into Your Application" section of the Inference Engine Development Guide [https://software.intel.com/inference-engine-devguide](https://software.intel.com/inference-engine-devguide)

#### Setting Up Command Line to Use OpenVINO Executables and Libraries

Whenever running the OpenVINO tools, compiling, or running the user application, always remember to source the script:

```Bash
source /opt/intel/computer_vision_sdk/bin/setupvars.sh
```


This script sets up the executable and library paths along with environment variables used by the OpenVINO tools as well as this tutorial.

### Where Do the Inference Models Come from?

An inference model may come from any of the supported sources and workflows such as Caffe, TensorFlow, and Apache MXNet.  For this tutorial, we will use models that have already been compiled by the Model Optimizer into .bin and .xml files and supplied within the OpenVINO toolkit samples.  The development and compiling of models is beyond the scope of this tutorial, for more information see [https://software.intel.com/openvino-toolkit/deep-learning-cv](https://software.intel.com/en-us/openvino-toolkit/deep-learning-cv)

# Key Concepts

Before going into the samples in the tutorial steps, first we will go over some key concepts that will be covered.

## Intel OpenCV

For the application that we will cover in Step 1, the OpenCV libraries included in the Intel CV SDK will be used.  You may be wondering: Why is OpenCV included in the CV SDK along with the Inference Engine?  The first big reason is: They are the fastest for Intel devices.  The Intel libraries have been optimized to run on each Intel CPU, GPU, and Myriad device.  It also helps that by including in the libraries in the CV SDK, you get the complete set of libraries and always get the necessary version.  

The second big reason: All the extensions and additional libraries that come with Intel’s OpenCV.  One such library is the Photography Vision Library (PVL).  PCL includes advanced implementations by Intel already optimized for power and performance on Intel devices to do face, blink, and smile detection along with recognizing faces.

More detail on the PVL library provided with Intel OpenCV may be found at:
[https://software.intel.com/en-us/cvsdk-devguide-advanced-face-capabilities-in-intels-opencv](https://software.intel.com/en-us/cvsdk-devguide-advanced-face-capabilities-in-intels-opencv)

## Floating Point Precision

Very briefly, floating point is a way to represent a wide range of real numbers with fraction within a fixed number of bits.  The upside to floating point is that a much larger range of numbers can be represented by the fewer bits, while the downside is that some amount of precision may be lost.  Here we we are talking about floating point being represented in either 32-bit, also referred to as "single-precision" and here we use “FP32”, or 16-bits, also referred to “half-precision” and here we use “FP16”.  Without going down to the bit-level details, just from the number of bits we can presume that 32-bits can represent more numbers than 16-bits.  

The question now becomes: why does 32 vs 16 bits matter?  First, because when a model’s IR is created using Intel’s Model Optimizer, it is told to target either FP16 or FP32.    For our purposes we will assume the model will work well as intended within the number of bits available (there may be differences in output of course, which are assumed to be small enough to ignore).  So now it comes down to what precision(s) are supported by the device that the model will be run on. The precisions that the CPU, GPU, and Myriad devices support is summarized in the table below:

<table>
  <tr>
    <td>Device</td>
    <td>FP16</td>
    <td>FP32</td>
  </tr>
  <tr>
    <td>CPU</td>
    <td>Not Supported</td>
    <td>Supported</td>
  </tr>
  <tr>
    <td>GPU</td>
    <td>Supported</td>
    <td>Supported</td>
  </tr>
  <tr>
    <td>Myriad</td>
    <td>Supported</td>
    <td>Not Supported</td>
  </tr>
</table>


### Why Would We Choose One Precision Over the Other?  

Primarily for speed.  FP16 is generally smaller and faster in hardware than FP32 while the smaller data size can also reduce by up to 2x the amount of memory required for storage and the bandwidth required for transferring data to and from memory.  However if the difference between FP32 and FP16 does affect the application’s results enough, then FP32 would be chosen to prioritize accuracy over speed.  Lastly, we would always choose FP16 vs. FP32 depending upon which precision the targeted device requires.

### What If We Specify the Wrong Precision for a Device?  

Nothing really will happen other than getting an error message when loading the model’s IR using the Inference Engine API which look similar to below:

```bash
[ INFO ] Loading Face Detection model to the MYRIAD plugin
[ ERROR ] The plugin does not support networks with FP32 format.
Supported format: FP16.
```


Of course, you will need to change to a supported precision for the device to be able to run the model.

In Step 4 we will see how precision may affect performance as well as device selection.

For more detail on 32-bit "single-precision" floating point down to the bit-level, see [https://en.wikipedia.org/wiki/Single-precision_floating-point_format](https://en.wikipedia.org/wiki/Single-precision_floating-point_format)

For more detail on 16-bit "half-precision" floating point down to the bit-level, see
[https://en.wikipedia.org/wiki/Half-precision_floating-point_format](https://en.wikipedia.org/wiki/Half-precision_floating-point_format)

For more detail on the Model Optimizer, see the OpenVINO documentation at:
[https://software.intel.com/openvino-toolkit/documentation](https://software.intel.com/openvino-toolkit/documentation)

## Batch Size

Batch size refers to the number of input data to be inferred during a single inference run through the Inference Engine.  Things to be aware of the batch size for an inference model:

* How batch size is set:

    * The default setting is located in the model’s IR which is set either by:

        * The Model Optimizer command-line option when creating the IR 

        * Or from the original source (e.g. Caffe) in which can be read using the Inference Engine API 

    * May be set explicitly using the Inference Engine API setBatchSize() function (see InferenceEngine::ICNNNetwork class)

* Will act as a maximum for the model once loaded and limit the number of inputs that will be inferred for each submitted request to the Inference Engine API

In this tutorial, face detection is done frame-by-frame expecting few results so batch size is primarily about device support such as the Myriad which requires batch size set to 1.  In a later tutorial (be sure to look forward to the Car Detection Tutorial), batch size will be explored further to show how it affects latency and performance. 

For more information on an example of batch size effects on performance for clDNN running on GPU, see the whitepaper: [https://software.intel.com/en-us/articles/accelerating-deep-learning-inference-with-intel-processor-graphics](https://software.intel.com/en-us/articles/accelerating-deep-learning-inference-with-intel-processor-graphics)

## Tutorial Step 1: Create the Base OpenCV Application

![image alt text](./doc_support/step0_image_2.png)

The first tutorial will show how the Intel OpenCV libraries are used by an application.  We will see how the OpenCV functions are included in an application as they are used to get input from image files or a video camera connected to the UP Squared board and display the image data in a window. 

[Face Detection Tutorial Step 1](./step_1/Readme.md)

## Tutorial Step 2: Add the first Model, Face Detection 

![image alt text](./doc_support/step0_image_3.png)

The second tutorial takes the framework in Tutorial Step 1 and add face detection and labeling to processed images.  This step shows how an inference model has been added to use the Inference Engine to run the model on hardware.  We will also learn how to specify which device the model is run on: CPU, GPU,  the Myriad.  

[Face Detection Tutorial Step 2](./step_2/Readme.md)

## Tutorial Step 3: Add the Second Model, Age and Gender

![image alt text](./doc_support/step0_image_4.png)

The third tutorial step will show how a second model is added to the application by including a model that infers age and gender of the detected face output from the face detection model.  

[Face Detection Tutorial Step 3](./step_3/Readme.md)

## Tutorial Step 4: Add the Third Model, Head Pose 

![image alt text](./doc_support/step0_image_5.png)

To complete the application, the fourth tutorial step adds a third model that infers head pose based on the detected face.  The three models now create the full processing pipeline which is used to explore options of assigning models to different device plugins to see differences in performance.

[Face Detection Tutorial Step 4](./step_4/Readme.md)

# Conclusion

Congratulations! you have completed the Face Detection Tutorial.  After going through this entire tutorial and all of its steps, you have now seen:

* An overview of the Inference Engine

* The final application assembled in steps:

    * How to create a base application that uses OpenCV to perform image and video input and output.  

    * How to extend the application to use the Inference Engine and CNN models to process the images and detect faces.  

    * How to take the results of the first model’s analysis and use it as input to different models that would process the faces and infer the face’s age and gender as well as the head’s orientation.  

    * How to load the analysis models onto different devices to distribute the workload and find the optimal device to get the best performance from the models.

# References and More Information
OpenVINO main page: [https://software.intel.com/openvino-toolkit](https://software.intel.com/openvino-toolkit)
OpenVINO documentation page: [https://software.intel.com/openvino-toolkit/documentation](https://software.intel.com/openvino-toolkit/documentation)
Deep Learning Deployment Toolkit: [https://software.intel.com/openvino-toolkit/deep-learning-cv](https://software.intel.com/openvino-toolkit/deep-learning-cv)

