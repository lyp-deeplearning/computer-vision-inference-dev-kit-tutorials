# Step 3: Add a second model, Age and Gender Detection

![image alt text](../doc_support/step3_image_0.png)

# Introduction

Welcome to Face Detection Tutorial Step 3.  Now that our application can detect faces in images, we now want our application to estimate the age and gender for each face.  The precompiled "age-gender-recognition-retail-0013" model included with OpenVINO that we will be running was trained on approximately 20,000 faces.  When it sees a face within 45 degrees (left, right, above, or below) of straight-on, it is 96.6% accurate on determining gender.  It can also determine ages to within 6 years, on average.  A sample output showing the results where the ROI box is now labeled “[M|F],<age>” appears below.  The metrics reported now also include the time to run the age and gender model.

![image alt text](../doc_support/step3_image_1.png)

# Age and Gender Detection Model

The Intel CV SDK provides a pre-compiled model for estimating age and gender from an image of a face.  You can find it at:

* /opt/intel/computer_vision_sdk/deployment_tools/intel_models/age-gender-recognition-retail-0013

The results it is capable of producing are shown in the summary below (for more details, see the descriptions HTML pages for each model): 

<table>
  <tr>
    <td>Model</td>
    <td>GFLOPS</td>
    <td>MParameters</td>
    <td>Average Precision</td>
  </tr>
  <tr>
    <td>age-gender-recognition-retail-0013</td>
    <td>0.094</td>
    <td>2.138</td>
    <td>Avg. age error: 6.07 years
Gender accuracy: 96.66%</td>
  </tr>
</table>


# Adding the Age and Gender Detection Model

Thanks to the setup work done in Tutorial Step 2, adding the age and gender detection model in this step will just be a matter of deriving a new class from the BaseDetection class, adding an additional command line argument to specify the new model, and updating the application to run and  track the statistics for the new model.  This means there will not be as much code to walk through this time.  That will let us focus on how to pass the important image inference results from the face detection model to the age and gender detection model.

1. Open up an Xterm window or use an existing window to get to a command shell prompt.

2. Change to the directory containing Tutorial Step 3:

```bash
cd tutorials/face_detection_tutorial/step_3
```


3. Open the files "main.cpp" and “face_detection.hpp” in the editor of your choice such as ‘gedit’, ‘gvim’, or ‘vim’.

4. The first section of code we want to look at is where we derive the AgeGenderDetection class from BaseDetection, and declare the member variables it uses.

```cpp
struct AgeGenderDetection : BaseDetection {
    std::string input;
    std::string outputAge;
    std::string outputGender;
    int enquedFaces = 0;
```


5. A difference to note is that when we use the default constructor from the BaseDetector class, we now pass in the batch size argument (FLAGS_n_ag) in addition to the age and gender model and the name we want to appear in printed messages.

```cpp
    AgeGenderDetection() : BaseDetection(FLAGS_m_ag, "Age Gender", FLAGS_n_ag) {}
```


```Cpp
    struct Result { float age; float maleProb;};
```


1. We also want to create our own Result class, to store the information that the model will return to us.  Namely, the age of the face and the probability that it is a male or female face.

```cpp
    Result operator[] (int idx) const {
        auto  genderBlob = request->GetBlob(outputGender);
        auto  ageBlob    = request->GetBlob(outputAge);

        return {ageBlob->buffer().as<float*>()[idx] * 100,
                genderBlob->buffer().as<float*>()[idx * 2 + 1]};
    }
```


2. Now we define the operator[] function, to give us an easy way to retrieve the age and gender results from the data contained in the output blob that we got after running inferrence.  To do that, we get index to the appropriate locations in the blob.  Then we return a result object containing the data read from the offset we calculate based on the index of the result.

## submitRequest()

```cpp
    void submitRequest() override {
        if (!enquedFaces) return;
        BaseDetection::submitRequest();
        enquedFaces = 0;
    }
```


For submitRequest(), we check to make sure that there are faces queued up to be processed.  If so, we call the base class submitRequest() function to tell the age and gender model to start performing inferences on the faces.  Then we set enquedFaces to 0, to indicate that all the faces have been submitted.

## enqueue()

```cpp
    void enqueue(const cv::Mat &face) {
        if (!enabled()) {
            return;
        }
        if (enquedFaces == maxBatch) {
            slog::warn << "Number of detected faces more than maximum(" << maxBatch << ") processed by Age Gender detector" << slog::endl;
            return;
        }
```


1. For the enqueue() function, the first things we need to do are to make sure that the model is enabled and make sure we have not reached the maximum number of faces that we can process.  If we have, then we log the warning and return without adding the face to the queue.

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


3. In this section of code, we get an input blob to hold the face input.  Then we convert the resulting face data that we get from the FaceDetection object and convert it to a blob that the age and gender model can work with.  On success, we increment the number of faces in the queue.

## read()

```cpp
    CNNNetwork read() override {
        slog::info << "Loading network files for AgeGender" << slog::endl;
        InferenceEngine::CNNNetReader netReader;
        /** Read network model **/
        netReader.ReadNetwork(FLAGS_m_ag);
```


1. In this section of code, we define the read() function that we override and customize for the needs of the age and gender model.  The first difference is that we get the name of the model from the FLAGS_m_ag command line parameter.

```cpp
        /** Set batch size to 16 **/
        netReader.getNetwork().setBatchSize(maxBatch);
        slog::info << "Batch size is set to " << netReader.getNetwork().getBatchSize() << " for Age Gender" << slog::endl;
```


2. We also set the maximum batch size to maxBatch (set using FLAGS_n_ag which defaults to 16) which can be >1 since the model was created to be able to make inferences on multiple faces at a time.

```cpp
        /** Extract model name and load it's weights **/
        std::string binFileName = fileNameNoExt(FLAGS_m_ag) + ".bin";
        netReader.ReadWeights(binFileName);
```


3. Again, we read in the IR .bin file of the model.  Notice that we do not try to read in labels for this model since it reports age and gender as values.

```cpp
        slog::info << "Checking Age Gender inputs" << slog::endl;
        InferenceEngine::InputsDataMap inputInfo(netReader.getNetwork().getInputsInfo());
        if (inputInfo.size() != 1) {
            throw std::logic_error("Age gender topology should have only one input");
        }
```


4. We do check for the proper number of inputs, and make sure that it has only the one input that we will use to pass in the face to process.

```cpp
        auto& inputInfoFirst = inputInfo.begin()->second;
        inputInfoFirst->setPrecision(Precision::FP32);
        inputInfoFirst->getInputData()->setLayout(Layout::NCHW);
        input = inputInfo.begin()->first;
```


5. We configure the input precision and memory layout, and save the name of the input blob for later use when getting a blob for input data.

```cpp
        slog::info << "Checking Age Gender outputs" << slog::endl;
        InferenceEngine::OutputsDataMap outputInfo(netReader.getNetwork().getOutputsInfo());
        if (outputInfo.size() != 2) {
            throw std::logic_error("Age Gender network should have two output layers");
        }
        auto it = outputInfo.begin();
        auto ageOutput = (it++)->second;
        auto genderOutput = (it++)->second;
```


6. Next, we verify that the model has the two output layers we expect for the age results and the gender results.  If it does, then we create and initialize our variables to hold the results we receive from the model.

```cpp
        // if gender output is convolution, it can be swapped with age
        if (genderOutput->getCreatorLayer().lock()->type == "Convolution") {
            std::swap(ageOutput, genderOutput);
        }

        if (ageOutput->getCreatorLayer().lock()->type != "Convolution") {
            throw std::logic_error("In Age Gender network, age layer (" + ageOutput->getCreatorLayer().lock()->name +
                ") should be a Convolution, but was: " + ageOutput->getCreatorLayer().lock()->type);
        }

        if (genderOutput->getCreatorLayer().lock()->type != "SoftMax") {
            throw std::logic_error("In Age Gender network, gender layer (" + genderOutput->getCreatorLayer().lock()->name +
                ") should be a SoftMax, but was: " + genderOutput->getCreatorLayer().lock()->type);
        }
```


7. We check the type of lock used by the model uses for the output layers.  If we find that the lock is "Convolution" then we swap the data to compensate for the layer reversal.  Then we check each output layer to make sure that it has the lock we expect it to.

```cpp
        slog::info << "Age layer: " << ageOutput->getCreatorLayer().lock()->name<< slog::endl;
        slog::info << "Gender layer: " << genderOutput->getCreatorLayer().lock()->name<< slog::endl;

        outputAge = ageOutput->name;
        outputGender = genderOutput->name;
```


8. Now we log the names of the two output layers and save them into our variables.

```cpp
        slog::info << "Loading Age Gender model to the "<< FLAGS_d_ag << " plugin" << slog::endl;
        _enabled = true;
        return netReader.getNetwork();
    }
};
```


9. Finally we log that we loaded the model, enable the model, and return the InferenceEngine::CNNNetwork object containing the model.

# Using AgeGenderDetection

That takes care of the new functions we defined to customize the BaseDetector class into the  AgeGenderDetection class specialized for the age and gender model.  We move down into the main function and see what additions have been made to apply the age and gender model to process the results of the face detection model.

1. First, we move to the section that loads the device plugins and update the cmdOptions to check for the additional command line arguments FLAGS_d_ag and FLAGS_m_ag.

```cpp
std::vector<std::pair<std::string, std::string>> cmdOptions = {
   {FLAGS_d, FLAGS_m}, {FLAGS_d_ag, FLAGS_m_ag}
};
```


2. We instantiate our age and gender detection object.

```cpp
AgeGenderDetection AgeGender;
```


3. The code that reads the command line and parses the command line options does not change.  Since we added the new arguments to the command line options vector, it just loops a second time to read in the flags for the new model we specified.  All we need to do is add a line of code to load in the new model and associate it with our AgeGender object.

```cpp
Load(AgeGender).into(pluginsForDevices[FLAGS_d_ag]);
```


4. Now, we move down into the "while(true)" loop and find where we retrieve the results from the face detection model.  We create a loop that iterates through the results.

```cpp
FaceDetection.fetchResults();
for (auto && face : FaceDetection.results) {
```


    1. We check to see if we want to use the age and gender model.  Then we combine the location of the face with its height and width to define a clipping rectangle. We apply that to the image frame to crop out just the face.

```cpp
   if (AgeGender.enabled()) {
      auto clippedRect = face.location & cv::Rect(0, 0, width, height);
      auto face = frame(clippedRect);
```


    2. Then we have the AgeGender object queue up each face for processing.

```cpp
      if (AgeGender.enabled()) {
         AgeGender.enqueue(face);
      }
   }
}
```


5. Once all of the input faces have been queued, we run the inference model while measuring the time it takes.

```cpp
t0 = std::chrono::high_resolution_clock::now();
if (AgeGender.enabled()) {
   AgeGender.submitRequest();
   AgeGender.wait();
}
t1 = std::chrono::high_resolution_clock::now();
ms secondDetection = std::chrono::duration_cast<ms>(t1 - t0);
```


6. Again, we output the timer results for inference adding the results for the age and gender analysis to the output window.

```cpp
if (AgeGender.enabled()) {
   out.str("");
   out << (AgeGender.enabled() ? "Age Gender"  : "")
       << "time: "<< std::fixed << std::setprecision(2) << secondDetection.count()
       << " ms ";
   if (!FaceDetection.results.empty()) {
      out << "(" << 1000.f / secondDetection.count() << " fps)";
   }
   cv::putText(frame, out.str(), cv::Point2f(0, 65), cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(255, 0, 0));
}
```


7. It is time to update the output image label with the results.  The first step is to get the location of the face.

```cpp
int i = 0;
for (auto & result : FaceDetection.results) {
   cv::Rect rect = result.location;
```


    3. Now we create a blank string to hold the results for the face and populate it with the results.

```cpp
   out.str("");
   if (AgeGender.enabled() && i < AgeGender.maxBatch) {
      out << (AgeGender[i].maleProb > 0.5 ? "M" : "F");
      out << std::fixed << std::setprecision(0) << "," << AgeGender[i].age;
   } else {
      out << (result.label < FaceDetection.labels.size() ? FaceDetection.labels[result.label] :
      std::string("label #") + std::to_string(result.label))
          << ": " << std::fixed << std::setprecision(3) << result.confidence;
   }
```


    4. Once the results string is populated, we can print it on the image data. 

```cpp
  cv::putText(frame,
               out.str(),
               cv::Point2f(result.location.x, result.location.y - 15),
               cv::FONT_HERSHEY_COMPLEX_SMALL,
               0.8,
               cv::Scalar(0, 0, 255));
```


    5. Now, we create a color based on the age and gender model’s confidence that the face is male.

```cpp
   auto genderColor =
         (AgeGender.enabled() && (i < AgeGender.maxBatch)) ?
            ((AgeGender[i].maleProb < 0.5) ? cv::Scalar(147, 20, 255) : cv::Scalar(255, 0, 0)) :
            cv::Scalar(100, 100, 100);
```


    6. Finally, we draw the rectangle on the image data and repeat the loop for the remaining faces in the image.

```   cv::rectangle(frame, result.location, genderColor, 1);
   i++;
}
```


8. At this point, we are at the bottom of the "while(true)" loop and all the image inference has been completed.  All that is left to do is output the final results for the frame while measuring the time it took to show the image.

```cpp
t0 = std::chrono::high_resolution_clock::now();
cv::imshow("Detection results", frame);
t1 = std::chrono::high_resolution_clock::now();
ocv_render_time = std::chrono::duration_cast<ms>(t1 - t0).count();
```


9. As in the previous examples, if the application was processing a single image or it has come to the final frame of the video file, it will wait for a keypress in the image window to exit.  If the application is processing images from a camera or there are more frames in the video file, it will loop back to the beginning and process another frame.

10. After all the images have been processed (or you have chosen to stop analyzing input from the camera), if the "-pc" command line argument was used, we print out the final performance statistics for all the models to the command window and exit the application.

```cpp
if (FLAGS_pc) {
   FaceDetection.printPerformanceCounts();
   AgeGender.printPerformanceCounts();
}
```


# Building and Running

We will now run our application and see how it performs using two models to process image data.  The complete code for the sample is in the Tutorial Step 3 directory.

1. Open up an Xterm window or use an existing window to get to a command shell prompt.

2. Change to the directory containing Tutorial Step 3:

```bash
cd tutorials/face_detection_tutorial/step_3
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


6. Before running, be sure to source the helper script that will make it easier to use environment variables instead of long names to the models:

```bash
source ../../scripts/setupenv.sh 
```


7. That should have created the executable for our application.  The only thing left to do is to tell you what to add to the command line, in order to load the age and gender model.  That is the "-m_ag" flag, followed by the full path to the model.

8. Before running, be sure to source the helper script that will make it easier to use environment variables instead of long names to the models:

```bash
source ../../scripts/setupenv.sh 
```


9. First let us see how it works on a single image file:

```bash
./intel64/Release/face_detection_tutorial -m $mFDA32 -m_ag $mAG32 -i ../../data/face.jpg
```


10. The output window should show the image overlaid with colored rectangles over each of the detected faces with labels showing the age and gender results.  The timing statistics for computing the results of each model along with OpenCV input and output times are also shown.

11. Next, let us try it on a video file.

```bash
./intel64/Release/face_detection_tutorial -m $mFDA32 -m_ag $mAG32 -i /opt/intel/computer_vision_sdk/openvx/samples/samples/face_detection/face.mp4
```


12. You should again see rectangles that now follow the faces around the image (if the faces move), accompanied by age and gender results for the faces, and the timing statistics for processing each frame of the video.

13. Finally, let us see how it works for camera input.

```bash
./intel64/Release/face_detection_tutorial -m $mFDA32 -m_ag $mAG32 -i cam
```


Or

```bash
./intel64/Release/face_detection_tutorial -m $mFDA32 -m_ag $mAG32
```


14. Again, you should see colored rectangles drawn around any faces that appear in the images, along with the results for age, gender, and the various render statistics.

15. When you press a key to exit the application, if you used the "-pc" parameter, the console window will be updated with the overall time statistics for processing all the frames.  This time, the output will show two groups of statistics, for the face detection model and the age and gender estimation model.  

# Conclusion

Building on the single model application from Tutorial Step 2, this step has shown that using a second inference model in an application is just as easy as using the first.  This also shows how powerful your applications can become by using one model to analyze the results you obtain from another model.  And that is the power OpenVINO brings to applications.  Continuing to Tutorial Step 4, we will expand the application once more by adding another model to estimate head pose based on the same face data that we used in Tutorial Step 3 to estimate age and gender.

# Navigation
[Face Detection Tutorial](../Readme.md)

[Face Detection Tutorial Step 2](../step_2/Readme.md)

[Face Detection Tutorial Step 4](../step_4/Readme.md)