# Step 4: Adding a third model, Head Pose Estimation

![image alt text](../doc_support/step4_image_0.png)

# Introduction

In Face Detection Tutorial Step 4, we will be including a final inference model.  This model estimates the head pose based the faces it is given.  For input, we will be using the same detected face results from the face detection model that we used in Tutorial Step 3, for the age and gender model.  After the head pose model has processed the face, it will draw a set of axes over the face, indicating the Yaw, Pitch, and Roll orientation of the head.  A sample output showing the results where the three axes appears below.  The metrics reported now also include the time to run the head pose model.

![image alt text](../doc_support/step4_image_1.png)

In the image above, the three axes intersect in the center of the head.  The blue line represents Roll, and it extends from the center of the head to the front and the back of the head.  The red line represents Pitch, and is drawn from the center of the head to the left ear.  The green line represents Yaw, and is drawn from the center of the head to the top of the head.

# Head Pose Estimation Model

The Intel CV SDK includes a pre-compiled model for estimating head pose from an image of a face.  You can find it at:

* /opt/intel/computer_vision_sdk/deployment_tools/intel_models/head-pose-estimation-adas-0001

The results it is capable of producing are shown in the summary below (for more details, see the descriptions HTML pages for each model): 

<table>
  <tr>
    <td>Model</td>
    <td>GFLOPS</td>
    <td>MParameters</td>
    <td>Average Precision</td>
  </tr>
  <tr>
    <td>head-pose-estimation-adas-0001</td>
    <td>0.0343</td>
    <td>0.92</td>
    <td>Angle: Mean ± standard deviation of absolute error
Yaw: 5.4 ± 4.4
Pitch: 5.5 ± 5.3
Roll: 4.6 ± 5.6</td>
  </tr>
</table>


# Adding the Head Pose Estimation Model

As we saw in Tutorial Step 3, adding a new model is a relatively straight forward process.  We just need to derive a new class for head pose estimation, add a new command line parameter, and add the necessary code to send the face data to the new model and retrieve the results.  Then we will take those results and overlay them on the face.  Let us take a look at the source code we use to accomplish that.

1. Open up an Xterm window or use an existing window to get to a command shell prompt.

2. Change to the directory containing Tutorial Step 4:

```bash
cd tutorials/face_detection_tutorial/step_4
```


3. Open the files "main.cpp" and “face_detection.hpp” in the editor of your choice such as ‘gedit’, ‘gvim’, or ‘vim’.

4. The first portion of the code we want to look at is where we derive the new HeadPoseDetection class from BaseDetection, and define the class functions we will be using.

```cpp
struct HeadPoseDetection : BaseDetection {
    std::string input;
    std::string outputAngleR = "angle_r_fc";
    std::string outputAngleP = "angle_p_fc";
    std::string outputAngleY = "angle_y_fc";
    int enquedFaces = 0;
    cv::Mat cameraMatrix;
```


5. You can see that we derive our HeadPoseDetection class from BaseDetection and add new variables specific to the head poses.  We will use strings to hold the model input, and the model output for head roll, pitch and yaw.  We also create a variable to let us know how many faces are queued up to be analyzed by the model.

```cpp
    HeadPoseDetection() : BaseDetection(FLAGS_m_hp, "Head Pose", FLAGS_n_hp) {}
```


6. Next, we call the constructor and let it store the Head Pose model, batch size, and the name we will use during output and logging.

```cpp
    struct Results {
        float angle_r;
        float angle_p;
        float angle_y;
    };
```


7. We need to define a Results structure to hold the data we get back from the head pose model.  It needs to hold values for the head roll, pitch and yaw.

```cpp
    Results operator[] (int idx) const {
```


8. Now we create an operator[] function for the Results class.  This gives us an easy way to access a specific set of angles in the data returned from the head pose model.

    ```cpp
            auto  angleR = request->GetBlob(outputAngleR);
            auto  angleP = request->GetBlob(outputAngleP);
            auto  angleY = request->GetBlob(outputAngleY);
    ```
    

    1. The data from the model is returned in 3 blobs, so we define three vectors that point to the data portion of the blobs.

    ```cpp
            return {angleR->buffer().as<float*>()[idx],
                    angleP->buffer().as<float*>()[idx],
                    angleY->buffer().as<float*>()[idx]};
        }
    ```
    

    2. Now that we have access to the three data vectors, we can return the values we need, based on their index into the vector.

## submitRequest()

```cpp
    void submitRequest() override {
        if (!enquedFaces) return;
        BaseDetection::submitRequest();
        enquedFaces = 0;
    }
```


For submitRequest(), we check to make sure that there are faces queued up to be processed.  If so, we call the base class submitRequest() function to tell the head pose model to start performing inferences on the faces.  Then we set enquedFaces to 0, to indicate that all the faces have been submitted.

## enqueue()

```cpp
   void enqueue(const cv::Mat &face) {
        if (!enabled()) {
            return;
        }
        if (enquedFaces == maxBatch) {
            slog::warn << "Number of detected faces more than maximum(" << maxBatch << ") processed by Head Pose detector" << slog::endl;
            return;
        }
```


1. For the enqueue() function,  we have to make sure that the head pose model is enabled and that we have not queued up more faces than the maximum batch size.

```cpp
        if (!request) {
            request = net.CreateInferRequestPtr();
        }
```


2. If we have not created a request object before, if not we will create one.

```cpp
        auto  inputBlob = request->GetBlob(input);

        if (matU8ToBlob<float>(face, inputBlob, 1.0f, enquedFaces)) {
        	enquedFaces++;
        }
    }
```


1. In this section of code, we get an input blob to hold the face input.  Then we convert the resulting face data that we get from the FaceDetection object and convert it to a blob that the head pose model can work with.  On success, we increment the number of faces in the queue.

## read()

```cpp
    CNNNetwork read() override {
```


1. Once again, we define the function that reads the model file and verifies that it uses the inputs and outputs that we expect it to.

```cpp
        slog::info << "Loading network files for Head Pose detection " << slog::endl;
        InferenceEngine::CNNNetReader netReader;
```


2. We use the InferenceEngine to create the netReader object

```cpp
        /** Read network model **/
        netReader.ReadNetwork(FLAGS_m_hp);
        /** Set batch size to maximum currently set to one provided from command line **/
        netReader.getNetwork().setBatchSize(maxBatch);
        slog::info << "Batch size is set to  " << netReader.getNetwork().getBatchSize() << " for Head Pose Network" << slog::endl;
```


3. Next, we copy the command line parameters into the netReader so it will read the model IR files.  We also configure the maximum batch size using maxBatch (set from the command line argument FLAGS_n_hp).

```cpp
        std::string binFileName = fileNameNoExt(FLAGS_m_hp) + ".bin";
        netReader.ReadWeights(binFileName);
```


4. After that, we can read the model bin file and get the weight information.

```cpp
        slog::info << "Checking Head Pose Network inputs" << slog::endl;
        InferenceEngine::InputsDataMap inputInfo(netReader.getNetwork().getInputsInfo());
        if (inputInfo.size() != 1) {
            throw std::logic_error("Head Pose topology should have only one input");
        }
        auto& inputInfoFirst = inputInfo.begin()->second;
        inputInfoFirst->setPrecision(Precision::FP32);
        inputInfoFirst->getInputData()->setLayout(Layout::NCHW);
        input = inputInfo.begin()->first;
```


5. We configure the input precision and memory layout, and save the name of the input blob for later use when getting a blob for input data.

```cpp
        slog::info << "Checking Head Pose network outputs" << slog::endl;
        InferenceEngine::OutputsDataMap outputInfo(netReader.getNetwork().getOutputsInfo());
        if (outputInfo.size() != 3) {
            throw std::logic_error("Head Pose network should have 3 outputs");
        }
        std::map<std::string, bool> layerNames = {
            {outputAngleR, false},
            {outputAngleP, false},
            {outputAngleY, false}
        };
```


6. The next step is to read the model and check out how many outputs it has.  It should have three, to return the roll, pitch, and yaw values.

```cpp
        for (auto && output : outputInfo) {
            auto layer = output.second->getCreatorLayer().lock();
            if (layerNames.find(layer->name) == layerNames.end()) {
                throw std::logic_error("Head Pose network output layer unknown: " + layer->name + ", should be " +
                    outputAngleR + " or " + outputAngleP + " or " + outputAngleY);
            }
            if (layer->type != "FullyConnected") {
                throw std::logic_error("Head Pose network output layer (" + layer->name + ") has invalid type: " +
                    layer->type + ", should be FullyConnected");
            }
            auto fc = dynamic_cast<FullyConnectedLayer*>(layer.get());
            if (fc->_out_num != 1) {
                throw std::logic_error("Head Pose network output layer (" + layer->name + ") has invalid out-size=" +
                    std::to_string(fc->_out_num) + ", should be 1");
            }
            layerNames[layer->name] = true;
        }
```


7. Finally, we need to verify that the three outputs have the correct names, layer types, and output size.  If the layer matches our requirements, then we set it "true" so we can use it.  

```cpp
        slog::info << "Loading Head Pose model to the "<< FLAGS_d_hp << " plugin" << slog::endl;

        _enabled = true;
        return netReader.getNetwork();
    }
```


8. Finally we log that we loaded the model, enable the model, and return the InferenceEngine::CNNNetwork object containing the model.

## buildCameraMatrix()

```cpp
buildCameraMatrix(int cx, int cy, float focalLength)
```


buildCameraMatrix() is a utility function that we use to create a "camera" that looks at the image.  It acts as the computer’s representation of “our eyes” during some calculations.

## drawAxes()

```cpp
drawAxes(cv::Mat& frame, cv::Point3f cpoint, Results headPose, float scale)
```


drawAxes() is another utility function that we use to create the Yaw, Pitch and Roll axes object that will be drawn on the screen.  It uses some standard math and trigonometry to determine how the three axes would look when viewed from the camera, and then draws the axes over the face.

# Using HeadPoseDetection

1. Jumping down to the main() function, we prepare the vector that stores the command line parameters to hold the new parameter and its value.

```cpp
std::vector<std::pair<std::string, std::string>> cmdOptions = {
   {FLAGS_d, FLAGS_m}, {FLAGS_d_ag, FLAGS_m_ag}, {FLAGS_d_hp, FLAGS_m_hp}
};
```


2. Then we create the new object to manage head pose functionality.

```cpp
HeadPoseDetection HeadPose;
```


3. With the new command line parameter added to the parser, we can load the head pose estimator model and associate it with the HeadPose object.

```cpp
Load(HeadPose).into(pluginsForDevices[FLAGS_d_hp]);
```


4. The next thing to do is to modify the loop that queues the faces to be analyzed by the age and gender model.  We just need to add a line that does a similar call to queue up the faces for the head pose model.  Here is the entire loop, for context.  You can also see that we are using the same input for both of the additional models.

```cpp
for (auto && face : FaceDetection.results) {
   if (AgeGender.enabled() || HeadPose.enabled()) {
      auto clippedRect = face.location & cv::Rect(0, 0, width, height);
      auto face = frame(clippedRect);
      if (AgeGender.enabled()) {
         AgeGender.enqueue(face);
      }
      if (HeadPose.enabled()) {
         HeadPose.enqueue(face);
      }
   }
}
```


5. Similarly, we look for the code we used in Tutorial Step 3 to have the AgeGender object analyze the faces, and we add the code to have the HeadPose object analyze the same faces.  We just have to make sure to add the new code before the code that records the time statistics for the image data analysis.  For context, here is that entire segment of code.

```cpp
t0 = std::chrono::high_resolution_clock::now();
if (AgeGender.enabled()) {
   AgeGender.submitRequest();
   AgeGender.wait();
}
if (HeadPose.enabled()) {
   HeadPose.submitRequest();
   HeadPose.wait();
}
t1 = std::chrono::high_resolution_clock::now();
ms secondDetection = std::chrono::duration_cast<ms>(t1 - t0);
```


6. With the image processing tasks done, it is time to update the output functionality to include the head pose results.

```cpp
if (HeadPose.enabled() || AgeGender.enabled()) {
   out.str("");
   out << (AgeGender.enabled() ? "Age Gender"  : "")
       << (AgeGender.enabled() && HeadPose.enabled() ? "+"  : "")
       << (HeadPose.enabled() ? "Head Pose "  : "")
       << "time: "<< std::fixed << std::setprecision(2) 
       << secondDetection.count()
       << " ms ";
   if (!FaceDetection.results.empty()) {
      out << "(" << 1000.f / secondDetection.count() << " fps)";
   }
   cv::putText(frame, out.str(), cv::Point2f(0, 65),
      cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(255, 0, 0));
}
```


7. We also update the section of code that creates the output image, so that it now adds draws the yaw, pitch, and roll axes over the faces.

```cpp
if (HeadPose.enabled() && i < HeadPose.maxBatch) {
   cv::Point3f center(rect.x + rect.width / 2, rect.y + rect.height / 2, 0);
   HeadPose.drawAxes(frame, center, HeadPose[i], 50);
}
```


8. The last thing we need to do it update the code to output the statistics for the head pose estimation model for when the -pc argument is used.

```cpp
if (FLAGS_pc) {
   FaceDetection.printPerformanceCounts();
   AgeGender.printPerformanceCounts();
   HeadPose.printPerformanceCounts();
}
```


# Building and Running

Now let us build and run the complete application and see how it runs all three analysis models.

1. Open up an Xterm window or use an existing window to get to a command shell prompt.

2. Change to the directory containing Tutorial Step 4:

```bash
cd tutorials/face_detection_tutorial/step_4
```


3. The first step is to configure the build environment for the OpenCV SDK by running the "setupvars.sh" script.

```bash
source  /opt/intel/computer_vision_sdk/bin/setupvars.sh
```


4. Now we need to create a directory to build the tutorial in and change to it.

```bash
mkdir build
cd build
```


5. The last thing we need to do before compiling is to configure the build settings and build the executable.  We do this by running CMake to set the build target and file locations.  Then we run Make to build the executable.

```bash
cmake -DCMAKE_BUILD_TYPE=Release ../
make
```


6. With the application built, all that is left is to run it using the a new "-m_hp" flag and the path to the head pose estimation model.

7. Before running, be sure to source the helper script that will make it easier to use environment variables instead of long names to the models:

```bash
source ../../scripts/setupenv.sh 
```


8. First, let us see how it works on a single image file.

```bash
./intel64/Release/face_detection_tutorial -m $mFDA32 -m_ag $mAG32 -m_hp $mHP32 -i ../../data/face.jpg
```


9. The output window will show the image overlaid with colored rectangles over the faces, age and gender results for each face, and the timing statistics for computing the results.  Additionally, you will see red, green, and blue axes over each face, representing the pose, or orientation, the head is in.

10. Next, let us try it on a video file.

```bash
./intel64/Release/face_detection_tutorial -m $mFDA32 -m_ag $mAG32 -m_hp $mHP32 -i /opt/intel/computer_vision_sdk/openvx/samples/samples/face_detection/face.mp4
```


11. You will see rectangles and the head pose axes that follow the faces around the image (if the faces move), accompanied by age and gender results for the faces, and the timing statistics for processing each frame of the video.

12. Finally, let us see how it works for camera input.  From the build directory, type 

```bash
./intel64/Release/face_detection_tutorial -m $mFDA32 -m_ag $mAG32 -m_hp $mHP32 -i cam
```


Or

```bash
./intel64/Release/face_detection_tutorial -m $mFDA32 -m_ag $mAG32 -m_hp $mHP32
```


13. Again, you will see colored rectangles drawn around any faces that appear in the images along with the results for age, gender, the axes representing the head poses, and the various render statistics.

14. When you press a key to exit the application, if you used the "-pc" parameter, the console window will be updated with the overall time statistics for processing all the frames.  This time, the output will show three groups of statistics, for the face detection model, the age and gender model, and the head pose model.  

# Picking the Right Models for the Right Devices

Throughout this tutorial, we just had our application load the models onto the default CPU device.  Here we will explore using the other devices included in the UP Squared AI Vision Development Kit, the GPU and Myriad.  That brings up several questions that we should discuss to get a more complete idea of how to make the best use of our models and how to optimize our applications using the devices available.

## What Determines the Device a Model Uses?

One of the main factors is going to be the floating point precision that the model requires.  We discuss that below, to answer the question "Are there models that cannot be loaded onto certain devices?"

Another major factor is speed.  Depending on how the model is structured, compiled and optimized, it may lend itself to running faster on a certain device.  Sometimes, you may know that.  Other times, you may have to test the model on different devices to determine where it runs best.

The other major factor in determining where to load a model is parallel processing or load balancing required to meet an application’s performance requirements.  

Once you have made those decisions, you can use the command line arguments to have the application assign the models to the particular device you want them to run on to test and verify.

## How Do I Choose the Specific Device to Run a Model?

In our application, we use command line parameters to specify which device to use for the models we load.  These are "-d", “-d_ag” and “-d_hp”, and they are used for the face detection model, age and gender estimation model, and head pose estimation model, respectively.  The available devices are “CPU”, “GPU” and “MYRIAD” that come with the UP Squared AI Vision Development Kit.

## Are There Models That Cannot be Loaded onto Specific Devices?

Yes.  The main restriction is the precision of the model must be supported by the device.  As we discussed in the Key Concepts section, certain devices can only run models that have the matching floating point precision.  For instance, the CPU can only run models that use FP32 precision.  This is because the hardware execution units of the CPU are designed and optimized for FP32 operations.  Similarly, the Myriad can only load models that use FP16 precision.  While the GPU is designed to be more flexible to run both FP16 and FP32 models, though it runs FP16 models faster than FP32 models.

## Are Some Devices Faster Than Others?

The easy answer is "yes."  The more complex answer is that it can be more complex than just “which device is fastest / has the fastest clock speed / and the most cores?”  Some devices are better at certain functions than other devices because of hardware optimizations or internal structures that fit the work being done within a particular model.  As noted previously, devices that can work with models running FP16 can run faster just because they are moving around as little as half the data of when running FP32.

## Are Some Devices Better for Certain Types of Models Than Other Devices?

Again, the easy answer is "yes."  The truth is that it can be difficult to know what model will run best on what device without actually loading the model on a device and seeing how it performs.  This is one of the most powerful features of the Inference Engine and Intel OpenVINO toolkit.  It is very easy to write applications that allow you to get up and running quickly to test many combinations of models and devices, without requiring significant code changes or even recompiling.  Our face detection application can do exactly that.  So let us see what we can learn about how these models work on different devices by running through the options.

### Command Line and All the Arguments

Before we can get started, let us go over the command line parameters again.  We specify the model we want to load by using the "-m*" arguments, which device to load using the “-d*” arguments, and batch size using the “-n*” arguments.  The table below summarizes the arguments for all three models.

<table>
  <tr>
    <td>Model</td>
    <td>Model Argument</td>
    <td>Device Argument</td>
    <td>Batch Size Argument</td>
  </tr>
  <tr>
    <td>Face detection</td>
    <td>-m</td>
    <td>-d</td>
    <td>(none, always set to 1)</td>
  </tr>
  <tr>
    <td>Age and gender</td>
    <td>-m_ag</td>
    <td>-d_ag</td>
    <td>-n_ag</td>
  </tr>
  <tr>
    <td>Head pose</td>
    <td>-m_hp</td>
    <td>-d_hp</td>
    <td>-n_hp</td>
  </tr>
</table>


As we mentioned in the Key Concepts section, the batch size is the amount of data that the models will work on.  For the face detection model, the batch size is always 1, because the model works on a single image at a time.  Even when processing input from a video or a camera, it only processes a single image/frame at a time.  Depending on the content of the image data, it can return any number of faces.  Our application lets us set the batch size on the other models dynamically and we have set a default maximum batch size of 16 for the age and gender and head pose models.  This is fine when they are run on the CPU and GPU, but is too much for the Myriad which requires batch size of 1.  So, when we want to run those two models on the Myriad, we will need to specify a maximum batch size of 1.

Let us look at a sample command line, that uses all the parameters, so we can see what it looks like.  For this example, we are running our application from the "step_4/build" directory.

```bash
./intel64/Release/face_detection_tutorial -m $mFDA32 -d GPU -m_ag $mAG16 -d_ag MYRIAD -n_ag 1 -m_hp $mHP16 -d_hp GPU -n_hp 16 -pc -i /opt/intel/computer_vision_sdk/openvx/samples/samples/face_detection/face.mp4
```


From this command line, we see that our application will load the FP32 face detection model onto the GPU, the FP16 age and gender model on the Myriad, using a batch size of 1, and the FP16 head pose model onto the GPU, with a batch size of 16.  We also specify "-pc" to capture the performance data, and “-i /opt/intel/computer_vision_sdk/openvx/samples/samples/face_detection/face.mp4” so that we have a “known” data set to do our performance tests with.  This MP4 video file used from the OpenVINO samples is a hand-drawn face with a moving camera.  

You can see that it is easy to change the model precision to match the device you want to run it on by changing the model to use the FP16 or FP32 using "16" and “32” built into the names of the variables..  It is easy to make up several test cases to see how our application and each of the inference model, perform.  Just remember that all models run on the CPU must be FP32, and all models run on the Myriad must be FP16.  Models run on the Myriad must also have their batch size set to 1.  Models run on the GPU can be either FP16 or FP32.

### What Kind of Performance Should I See?

That depends on many things, from the specific combination of models and devices that you specified, to the other applications running on the development kit while you collect data.  There are are also different versions of the UP Squared board, with different CPU and GPU hardware.  So the exact data will vary from what appears in the chart below, however the general trends should be the same.  That said, let us take a look at some of the performance counts we observed.  The performance data is recorded in microseconds of calculation time.  If you want to know how that converts to seconds, you can shift the decimal point six places to the left.  That means if the Face Detection model spends 171604 microseconds analyzing data, it is taking 0.171604 seconds.  Also remember that this is the amount of time spent processing the entire video, not for each frame.

Below are ten command lines we used to generate some performance count data.  They do not cover all the possible combinations, but they do give a good indication of the top performance trends for each of the models on the various devices.  Remember that when we load a model onto the Myriad, we have to use the associated -n* parameter to set the batch size to 1.

```bash
./intel64/Release/face_detection_tutorial -m $mFDA32 -d CPU -m_ag $mAG32 -d_ag CPU -m_hp $mHP32 -d_hp CPU -i /opt/intel/computer_vision_sdk/openvx/samples/samples/face_detection/face.mp4 -pc
./intel64/Release/face_detection_tutorial -m $mFDA32 -d CPU -m_ag $mAG32 -d_ag CPU -m_hp $mHP32 -d_hp GPU -i /opt/intel/computer_vision_sdk/openvx/samples/samples/face_detection/face.mp4 -pc
./intel64/Release/face_detection_tutorial -m $mFDA32 -d GPU -m_ag $mAG32 -d_ag CPU -m_hp $mHP32 -d_hp GPU -i /opt/intel/computer_vision_sdk/openvx/samples/samples/face_detection/face.mp4 -pc
./intel64/Release/face_detection_tutorial -m $mFDA16 -d GPU -m_ag $mAG32 -d_ag CPU -m_hp $mHP32 -d_hp CPU -i /opt/intel/computer_vision_sdk/openvx/samples/samples/face_detection/face.mp4 -pc
./intel64/Release/face_detection_tutorial -m $mFDA16 -d MYRIAD -m_ag $mAG16 -d_ag GPU -m_hp $mHP32 -d_hp CPU -i /opt/intel/computer_vision_sdk/openvx/samples/samples/face_detection/face.mp4 -pc
./intel64/Release/face_detection_tutorial -m $mFDA16 -d MYRIAD -m_ag $mAG16 -d_ag GPU -m_hp $mHP32 -d_hp GPU -i /opt/intel/computer_vision_sdk/openvx/samples/samples/face_detection/face.mp4 -pc
./intel64/Release/face_detection_tutorial -m $mFDA32 -d CPU -m_ag $mAG16 -d_ag GPU -m_hp $mHP16 -d_hp GPU /opt/intel/computer_vision_sdk/openvx/samples/samples/face_detection/face.mp4 -pc
./intel64/Release/face_detection_tutorial -m $mFDA32 -d CPU -m_ag $mAG16 -d_ag MYRIAD -n_ag 1 -m_hp $mHP16 -d_hp MYRIAD -n_hp 1 -i /opt/intel/computer_vision_sdk/openvx/samples/samples/face_detection/face.mp4 -pc
./intel64/Release/face_detection_tutorial -m $mFDA32 -d GPU -m_ag $mAG16 -d_ag MYRIAD -n_ag 1 -m_hp $mHP16 -d_hp MYRIAD -n_hp 1 -i /opt/intel/computer_vision_sdk/openvx/samples/samples/face_detection/face.mp4 -pc
./intel64/Release/face_detection_tutorial -m $mFDA16 -d GPU -m_ag $mAG16 -d_ag MYRIAD -n_ag 1 -m_hp $mHP16 -d_hp MYRIAD -n_hp 1 -i /opt/intel/computer_vision_sdk/openvx/samples/samples/face_detection/face.mp4 -pc
```


<table>
  <tr>
    <td></td>
    <td>Face Detection</td>
    <td>Age Gender</td>
    <td>Head Pose</td>
    <td>FD Time</td>
    <td>AG Time</td>
    <td>HP Time</td>
  </tr>
  <tr>
    <td>1</td>
    <td>CPU, FP32</td>
    <td>CPU, FP32</td>
    <td>CPU, FP32</td>
    <td>171604</td>
    <td>139568</td>
    <td>131307</td>
  </tr>
  <tr>
    <td>2</td>
    <td>CPU, FP32</td>
    <td>CPU, FP32</td>
    <td>GPU, FP32</td>
    <td>145526</td>
    <td>135789</td>
    <td>41434</td>
  </tr>
  <tr>
    <td>3</td>
    <td>GPU, FP32</td>
    <td>CPU, FP32</td>
    <td>CPU, FP32</td>
    <td>86002</td>
    <td>68433</td>
    <td>112021</td>
  </tr>
  <tr>
    <td>4</td>
    <td>GPU, FP16</td>
    <td>CPU, FP32</td>
    <td>CPU, FP32</td>
    <td>76740</td>
    <td>207186</td>
    <td>108561</td>
  </tr>
  <tr>
    <td>5</td>
    <td>MYRIAD, FP16</td>
    <td>GPU, FP16</td>
    <td>CPU, FP32</td>
    <td>333072</td>
    <td>32515</td>
    <td>62198</td>
  </tr>
  <tr>
    <td>6</td>
    <td>MYRIAD, FP16</td>
    <td>GPU, FP16</td>
    <td>GPU, FP32</td>
    <td>333193</td>
    <td>12068</td>
    <td>30891</td>
  </tr>
  <tr>
    <td>7</td>
    <td>CPU, FP32</td>
    <td>GPU, FP16</td>
    <td>GPU, FP16</td>
    <td>89942</td>
    <td>19223</td>
    <td>20418</td>
  </tr>
  <tr>
    <td>8</td>
    <td>CPU, FP32</td>
    <td>MYRIAD, FP16</td>
    <td>MYRIAD, FP16</td>
    <td>88223</td>
    <td>9902</td>
    <td>7533</td>
  </tr>
  <tr>
    <td>9</td>
    <td>GPU, FP32</td>
    <td>MYRIAD, FP16</td>
    <td>MYRIAD, FP16</td>
    <td>68073</td>
    <td>9723</td>
    <td>7814</td>
  </tr>
  <tr>
    <td>10</td>
    <td>GPU, FP16</td>
    <td>MYRIAD, FP16</td>
    <td>MYRIAD, FP16</td>
    <td>56315</td>
    <td>9779</td>
    <td>7706</td>
  </tr>
</table>


Above is a simple matrix that shows some of the test cases that we ran on a UP Squared Apollo Lake Intel Pentium N4200 to get performance data.  The analysis models are listed on the left, with their performance numbers on the right.  In each row, we list the device the model was run on and the floating point precision.  From this, we can see several trends.  FP32 performance is generally faster on the GPU than on the CPU.  FP16 performance is almost always faster than FP32 performance on any device.  And the Myriad is a perplexing device, as it shows the fastest and slowest performance numbers.  This is probably due to the architecture of the device and that is connected by USB for data transfers, instead of being more directly connected to memory like the CPU and GPU.  This means that more compute time needs to be spent moving data back and forth, across a slower connection.  So we can infer that models that require significant amounts of data to process would be better run on the CPU or GPU, while model that work on modest amounts of data transfer can be handled quite well by the Myriad.

In general, the best way to maximize performance is to put the most complex model on the fastest device.  Try to divide the work across the devices as much as possible to avoid overloading any one device.  If you do not need FP32 precision, you can speed up your applications by using FP16 models.

In our specific case, we see the best performance when face detection analysis is run on the GPU, using the FP16 model, and the age and gender and head pose models are run on the Myriad, using FP16 models and a maximum batch size of 1.  This works for this specific case, because we know that the video we are analyzing only has a single face in it.  If we were analyzing video with more than one face, the more optimal solution would be a between running FP32 face detection on the CPU and FP16 age and gender and head pose inference on the GPU, or running all three models on the GPU using FP16 models.

Something to note too, is that the Myriad is only capable of running two analysis models at a time.  If you try to load a third model, the application will exit, and report a "Device not found" error.

# Conclusion

By adding the head pose estimation model to the application from Tutorial Step 3, you have now seen the final step in assembling the full application.  This again shows the power OpenVINO brings to applications by quickly being able to add another inference model.  We also discussed how to load the inference models onto different devices to distribute the workload and find the optimal device to get the best performance from the models.

# Navigation
[Face Detection Tutorial](../Readme.md)

[Face Detection Tutorial Step 3](../step_3/Readme.md)