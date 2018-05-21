# Step 2: Add the first model, Face Detection

![image alt text](../doc_support/step2_image_0.png)

# Introduction

Welcome to the Face Detection Tutorial Step 2.  This is the step of the tutorial where it gets its name by processing image data and detecting faces.  We get this ability by having our application use the Inference Engine to load and run the Intermediate Representation (IR) of a CNN model on the selected hardware device CPU, GPU, or Myriad to perform face detection.  You may recall from the OpenVINO overview, an IR model is a compiled version of a CNN (e.g. from Caffe) that has been optimized using the Model Optimizer for use with the Inference Engine.  This is where we start to see the power of the OpenVINO toolkit to load and run models on devices.  In this tutorial step, we will use the Inference Engine to run a pre-compiled model to do face detection on the input image and then output the results.  

A sample output showing the results where a Region of Interest (ROI) box appears around the detected face below.  The metrics reported include the time for OpenCV capture and display along with the time to run the face detection model.  The detected face gets a box around it along with a label as shown below.

![image alt text](../doc_support/step2_image_1.png)

# Face Detection Models

The Intel CV SDK includes two pre-compiled face detection models located at:

* /opt/intel/computer_vision_sdk/deployment_tools/intel_models/face-detection-adas-0001

    * Available model locations:

        * FP16: /opt/intel/computer_vision_sdk/deployment_tools/intel_models/face-detection-adas-0001/FP16/face-detection-adas-0001.xml

        * FP32: /opt/intel/computer_vision_sdk/deployment_tools/intel_models/face-detection-adas-0001/FP32/face-detection-adas-0001.xml

    * More detail may be found in the CV SDK installation at:       file:///opt/intel/computer_vision_sdk/deployment_tools/intel_models/face-detection-adas-0001/description/face-detection-adas-0001.html

* /opt/intel/computer_vision_sdk/deployment_tools/intel_models/face-detection-retail-0004

    * Available model locations:

        * FP16: /opt/intel/computer_vision_sdk/deployment_tools/intel_models/face-detection-retail-0004/FP16/face-detection-retail-0004.xml

        * FP32: /opt/intel/computer_vision_sdk/deployment_tools/intel_models/face-detection-retail-0004/FP32/face-detection-retail-0004.xml

    * More detail may be found in the CV SDK installation at: file:///opt/intel/computer_vision_sdk/deployment_tools/intel_models/face-detection-retail-0004/description/face-detection-retail-0004.html

Each model may be used to perform face detection, the difference is how complex each underlying model is for the results it is capable of producing as shown in the summary below (for more details, see the descriptions HTML pages for each model): 

<table>
  <tr>
    <td>Model</td>
    <td>GFLOPS</td>
    <td>MParameters</td>
    <td>Average Precision</td>
  </tr>
  <tr>
    <td>face-detection-adas-0001</td>
    <td>1.4</td>
    <td>1.1</td>
    <td>AP (head height >64px) 93.1%
AP (head height >100px) 94.1%</td>
  </tr>
  <tr>
    <td>face-detection-retail-0004</td>
    <td>1.067079</td>
    <td>0.58822</td>
    <td>AP (images > 60x60px) 83.00%for </td>
  </tr>
</table>


Note that each model comes pre-compiled for both FP16 and FP32, when choosing which precision always be sure to make sure the hardware device supports it.   

## How Do I Specify Which Device the Model Will Run On?

To make it easier to try different assignments, we will make our application be able to set which device a model is to be run on using command line arguments.  The default device will be the CPU when not set.  Now we will do a brief walkthrough how this is done in the code starting from the command line arguments to the Inference Engine API calls.  Here we are highlighting the specific code, so some code will be skipped over for now to be covered later in other walkthroughs.

To create the command line arguments, we use the previously described gflags helper library to define the arguments for specifying both the face detection model and the device to run it on.  The code appears in "face_detection.hpp":

```cpp
/// @brief message for plugin argument
static const char plugin_message[] = "Plugin name. For example MKLDNNPlugin. If this parameter is pointed, " \
"the sample will look for this plugin only.";

/// \brief Define parameter for face detection  model file <br>
/// It is a required parameter
DEFINE_string(m, "", face_detection_model_message);
```


To create the command line argument: -m <model-IR-xml-file>, where <model-IR-xml-file> is the face detection model’s .xml file

```bash
/// @brief message for assigning face detection calculation to device
static const char target_device_message[] = "Specify the target device for Face Detection (CPU, GPU, FPGA, or MYRYAD. " \
"Sample will look for a suitable plugin for device specified.";

/// \brief device the target device for face detection infer on <br>
DEFINE_string(d, "CPU", target_device_message);
```


To create the argument: -d <device>, where <device> is set to "CPU", "GPU", or "MYRIAD" which we will see conveniently matches what will be passed to the Inference Engine later.

As a result of the macros used in the code above, the variables "FLAGS_m" and “FLAGS_d” have been created to hold the argument values.  Focusing primarily on how the “FLAGS_d” is used to tell the Inference Engine which device to use, we follow the code in “main()” of “main.cpp”:

1. First we declare a map to hold the plugins as they are loaded.  The mapping will allow the associated plugin InferencePlugin object to be found by name (e.g. "CPU") 
```cpp
     // ---------------------Load plugins for inference engine------------------------------------------------
        std::map<std::string, InferencePlugin> pluginsForDevices;
```

2. We use a vector to pair the device and model command line arguments so we can iterate through them:     

```cpp
   std::vector<std::pair<std::string, std::string>> cmdOptions = {
            {FLAGS_d, FLAGS_m}
        };
```


3. We iterate through device and model the argument pairs:

```cpp
for (auto && option : cmdOptions) {
            auto deviceName = option.first;
            auto networkName = option.second;
```


4. We make sure the plugin has not already been created and put into the pluginsForDevices map: 
        
```cpp
 if (pluginsForDevices.find(deviceName) != pluginsForDevices.end()) {
                continue;
            }
```

5. We create the plugin using the Inference Engine’s PluginDispatcher API for the given device’s name.  Here "deviceName" is the value for “FLAGS_d” which came directly from the command line argument “-d” which is set to “CPU”, “GPU”, or “MYRIAD”, the exact names the Inference Engine knows for devices.

```cpp
         slog::info << "Loading plugin " << deviceName << slog::endl;
            InferencePlugin plugin = PluginDispatcher({"../../../lib/intel64", ""}).getPluginByDevice(deviceName);
```
6. Report plugin details:

```cpp 
           /** Printing plugin version **/
            printPluginVersion(plugin, std::cout);
```
7. Store the created plugin to be found by device name later:

```cpp
            pluginsForDevices[deviceName] = plugin;
```
8. Finally load the model passing the plugin created for the specified device, again using the name given same as it is on the command line ("Load" class will be described later):

```cpp
       // --------------------Load networks (Generated xml/bin files)-------------------------------------------
        Load(FaceDetection).into(pluginsForDevices[FLAGS_d]);
```

### Verifying Which Device is Running the Model

Our application will give output saying what Inference Engine plugins (devices) were loaded and which models were loaded on which plugins.

Here is a sample of the output in the console window:

Inference Engine reporting its version:

```bash
InferenceEngine: 
	API version ............ 1.0
	Build .................. 10073
[ INFO ] Parsing input parameters
[ INFO ] Reading input
```
Our application reporting it is loading the CPU plugin:
```bash
[ INFO ] Loading plugin CPU
```
Inference Engine reports it has loaded the CPU plugin (MKLDNNPlugin) and its version:
```bash
	API version ............ 1.0
	Build .................. lnx_20180314
	Description ....... MKLDNNPlugin
[ INFO ] Loading network files for Face Detection
[ INFO ] Batch size is set to  1
[ INFO ] Checking Face Detection inputs
[ INFO ] Checking Face Detection outputs
```
Our application reporting it is loading the CPU plugin for the face detection model:
```bash
[ INFO ] Loading Face Detection model to the CPU plugin
[ INFO ] Start inference
```

Later in Tutorial Step 4, we cover loading multiple models onto different devices.  We will also look at how the models perform on different devices.  Until then, we will let all the models load and run on the default CPU device.

# Adding the Face Detection Model

From Tutorial Step 1, we have the base application that can read and display image data, now it is time process the images.  This step of the tutorial expands the capabilities of our application to use the Inference Engine and the face detection model to process images.  To accomplish this, first we are going to introduce some helper functions and classes.  The code may be found in the main.cpp file.

### Helper Functions and Classes

We are going to need a function that takes our input image and turns it into a "blob".  Which begs the question “What is a blob?”  In short, a blob, specifically the class InferenceEngine::Blob, is the data container type used by the Inference Engine for holding input and output data.  To get data into our model, we will need to first convert the image data from the OpenCV cv::Mat to an InferenceEngine::Blob.  To do that we have the provided helper function “matU8ToBlob” in main.cpp: 

#### matU8ToBlob

```cpp
// Returns 1 on success, 0 on failure
int matU8ToBlob(const cv::Mat& orig_image, Blob::Ptr& blob, float scaleFactor = 1.0, int batchIndex = 0)
{
    SizeVector blobSize = blob.get()->dims();
    const size_t width = blobSize[0];
    const size_t height = blobSize[1];
    const size_t channels = blobSize[2];
    T* blob_data = blob->buffer().as<T*>();
```


1. Here we define variables that store the dimensions for the images that the IR is optimized to work with.  Then we assign "blob_data" to the blob’s data buffer.

```cpp
    cv::Mat resized_image(orig_image);
    if (width != orig_image.size().width || height!= orig_image.size().height) {
    	   // ignore rectangles with either dimension < 1
    	   if (orig_image.size().width < 1 || orig_image.size().height < 1) {
    		return 0;
    	   }
        cv::resize(orig_image, resized_image, cv::Size(width, height));
    }
```


2. Next, we check to see if the input image matches the dimensions of images that the IR is expecting.  If the dimensions do not match, then we use the OpenCV functions to resize it.  We also make sure that a input with either height or width are <1 is not stored, returning 0 to indicate nothing was done.

```cpp
    int batchOffset = batchIndex * width * height * channels;

    for (size_t c = 0; c < channels; c++) {
        for (size_t  h = 0; h < height; h++) {
            for (size_t w = 0; w < width; w++) {
                blob_data[batchOffset + c * width * height + h * width + w] =
                    resized_image.at<cv::Vec3b>(h, w)[c] * scaleFactor;
            }
        }
    }
    return 1;
}
```


3. Now that we have the image data in the proper size, we copy the data from the input image into the blob’s buffer.  A blob will hold the entire batch for a run through the inference model, so for each batch item we first calculate "batchOffset" as an offset into the blob’s buffer before copying the data.

For more details on the InferenceEngine::Blob class, see "Inference Engine Memory primitives" in the documentation.

#### Load

The next helper function we are going to define loads the model onto the device we want it to execute on.  

```cpp
struct Load {
    BaseDetection& detector;
    explicit Load(BaseDetection& detector) : detector(detector) { }

    void into(InferenceEngine::InferencePlugin & plg) const {
        if (detector.enabled()) {
            detector.net = plg.LoadNetwork(detector.read(), {});
            detector.plugin = &plg;
        }
    }
};
```


To help explain how this works, an example using "Load" will be used which looks like:

```cpp
Load(FaceDetection).into(pluginsForDevices[FLAGS_d]);
```


Which is read as "Load FaceDetection into the plugin pluginsForDevices[FLAGS_d]" which is done as follows:

1. Load(FaceDetection) is a constructor to initialize model object "detector" and returns a “Load” object

2. "into()" is called on the returned object passing in the mapped plugin from “pluginsForDevices”.  The map returns the plugin mapped to “FLAGS_d”, which is the command line argument “CPU”, “GPU”, or “MYRIAD”.  The function into() then first checks if the model object is enabled and if it is:

    1. Calls "plg.LoadNetwork(detector.read(),{})"  to load the model returned by “detector.read()” (which we will see later is reading in the model’s IR file) into the plugin.  The resulting object is stored in the model object (detetor.net) 

    2. Sets the model object’s plugin (detector.plugin) to the one used

#### BaseDetection Class

Now we are going to walkthrough the BaseDetection class that we use to abstract common features and functionality when using a model which the code also refers to as "detector".  

1. Here we declare the class and define its member variables and the default constructor and destructor.

```cpp
struct BaseDetection {
    ExecutableNetwork net;
    InferenceEngine::InferencePlugin * plugin;
    InferRequest::Ptr request;
    std::string & commandLineFlag;
    std::string topoName;
    const int maxBatch;

    BaseDetection(std::string &commandLineFlag, std::string topoName, int maxBatch)
        : commandLineFlag(commandLineFlag), topoName(topoName), maxBatch(maxBatch) {}

    virtual ~BaseDetection() {}
```


2. The ExecutableNetwork holds the model that will be used to process the data and make inferences.  The InferencePlugin is the Inference Engine plugin that will be executing the Intermediate Reference on a specific device.  InferRequest is an object that we will use to tell the model that we want it to make inferences on some data.  The name of the model is stored in topoName and the command line parameters for this model are stored in commandLineFlag.  Finally, maxBatch is used to limit the number of items to that we will try to process, so we do not exceed the limits of the models or the hardware.

```cpp
    ExecutableNetwork* operator ->() {
        return &net;
    }
```


3. We override operator -> so that we have a convenient way to get access to the network.

```cpp
    virtual InferenceEngine::CNNNetwork read()  = 0;
```


4. Since the networks used by the detectors could have different requirements for loading, we declare the read() function to be pure virtual.  This ensures that each detector class will have a read function appropriate to the IR it will be using.

```cpp
    virtual void submitRequest() {
        if (!enabled() || request == nullptr) return;
        request->StartAsync();
    }
```


5. The submitRequest() function checks to see if the model is enabled and that there is a valid request to start.  If there is, it requests the model to start running the model asynchronously (we will show how to wait on the results next).

```cpp
    virtual void wait() {
        if (!enabled()|| !request) return;
        request->Wait(IInferRequest::WaitMode::RESULT_READY);
    }
```


6. We use wait() to wait until results from the model are ready.  Wait first checks to see if the model is enabled and there is a valid request before actually waiting on the request.

```cpp
    mutable bool enablingChecked = false;
    mutable bool _enabled = false;

    bool enabled() const  {
        if (!enablingChecked) {
            _enabled = !commandLineFlag.empty();
            if (!_enabled) {
                slog::info << topoName << " DISABLED" << slog::endl;
            }
            enablingChecked = true;
        }
        return _enabled;
    }
```


7. Here we define variables and the "enabled()" function to track and check if the model is enabled or not.  The model is disabled if “commandLineFlag”, the command line argument specifying the model IR .xml file (e.g. “-m”) , has not been set.

```cpp
    void printPerformanceCounts() {
        if (!enabled()) {
            return;
        }
        slog::info << "Performance counts for " << topoName << slog::endl << slog::endl;
        ::printPerformanceCounts(request->GetPerformanceCounts(), std::cout, false);
    }
```


8. Finally, the printPerformancCount() function checks to see if the detector is enabled, and if it is, then we print out the overall performance statistics for the model.

#### FaceDetectionClass 

Now that we have seen what the base class provides, we will now walkthrough the code for the derived FaceDetectionClass class to see how a model is implemented.

We will start by deriving it from the BaseDetection class and adding some new member variables that we will need.

```cpp
struct FaceDetectionClass : BaseDetection {
    std::string input;
    std::string output;
    int maxProposalCount;
    int objectSize;
    int enquedFrames = 0;
    float width = 0;
    float height = 0;
    bool resultsFetched = false;
    std::vector<std::string> labels;
    using BaseDetection::operator=;

    struct Result {
        int label;
        float confidence;
        cv::Rect location;
    };

    std::vector<Result> results;
```


Notice the "Result" struct and the vector “results” that will be used since the face detection model can find more than one face in an image.  Each Result will include a cvRect indicating the location and size of the face in the input image.  It will also have a label and a value indicating the confidence that the result is a face.

##### submitRequest()

```cpp
    void submitRequest() override {
        if (!enquedFrames) return;
        enquedFrames = 0;
        resultsFetched = false;
        results.clear();
        BaseDetection::submitRequest();
    }
```


We override the submitRequest() function to make sure there is input data ready and clear out any previous results before calling BaseDetection::submitRequest() to start inference.

##### enqueue()

```cpp
    void enqueue(const cv::Mat &frame) {
        if (!enabled()) return;

        if (!request) {
            request = net.CreateInferRequestPtr();
        }
```


We check to see that the face detection model is enabled.  If it has, then we create an inference request object if one has not been already.  The request object is is how we communicate with the model, sending and receiving data and starting inferences and waiting on completion.

```cpp
        width = frame.cols;
        height = frame.rows;

        auto  inputBlob = request->GetBlob(input);
        if (matU8ToBlob<uint8_t >(frame, inputBlob)) {
        	enquedFrames = 1;
        }
     }
```


Here, we get an input blob from the model and then use matU8ToBlob() to copy the image image data into the blob.

##### FaceDetectionClass()

```cpp
    FaceDetectionClass() : BaseDetection(FLAGS_m, "Face Detection", 1) {}
```


We call the base class constructor, passing in the model to load specified in the command line argument FLAGS_m, the name we want to call it when we print out informational messages, and set the batch size to 1.  This initializes the BaseDetection subclass specifically for FaceDetectionClass.

##### read()

The next function we will walkthrough is the FaceDetectorClass::read() function which must be specialized specifically to the model that it will load and run. 

```cpp
    InferenceEngine::CNNNetwork read() override {
        slog::info << "Loading network files for Face Detection" << slog::endl;
        InferenceEngine::CNNNetReader netReader;
        /** Read network model **/
        netReader.ReadNetwork(FLAGS_m);
```


1. We use the Inference Engine API InferenceEngine::CNNNetReader object to load the model IR files.  This comes from the XML file that we specified on the command line, using the "-m" parameter.  

```cpp
        /** Set batch size to 1 **/
        slog::info << "Batch size is set to  "<< maxBatch << slog::endl;
        netReader.getNetwork().setBatchSize(maxBatch);
```


2. We set the maximum batch size to 1 because the face detection models to be used are designed to infer one image at a time.

```cpp
        /** Extract model name and load it's weights **/
        std::string binFileName = fileNameNoExt(FLAGS_m) + ".bin";
        netReader.ReadWeights(binFileName);
        /** Read labels (if any)**/
        std::string labelFileName = fileNameNoExt(FLAGS_m) + ".labels";

        std::ifstream inputFile(labelFileName);
        std::copy(std::istream_iterator<std::string>(inputFile),
                  std::istream_iterator<std::string>(),
                  std::back_inserter(labels));
```


3. We generate names for the model IR .bin file and optional .label file based on the model name from the "-m" parameter.  

```cpp
        slog::info << "Checking Face Detection inputs" << slog::endl;
        InferenceEngine::InputsDataMap inputInfo(netReader.getNetwork().getInputsInfo());
        if (inputInfo.size() != 1) {
            throw std::logic_error("Face Detection network should have only one input");
        }
        auto& inputInfoFirst = inputInfo.begin()->second;
        inputInfoFirst->setPrecision(Precision::U8);
        inputInfoFirst->getInputData()->setLayout(Layout::NCHW);
```


4. Next, we prepare the input data format to configure it for the proper precision (U8 = 8-bit per BGR channel) and memory layout for the model (NCHW).  

```        slog::info << "Checking Face Detection outputs" << slog::endl;
        InferenceEngine::OutputsDataMap outputInfo(netReader.getNetwork().getOutputsInfo());
        if (outputInfo.size() != 1) {
            throw std::logic_error("Face Detection network should have only one output");
        }
```


5. We make sure that there is only one output result defined for the model.

```cpp
        auto& _output = outputInfo.begin()->second;
        output = outputInfo.begin()->first;

        const auto outputLayer = netReader.getNetwork().getLayerByName(output.c_str());
        if (outputLayer->type != "DetectionOutput") {
            throw std::logic_error("Face Detection network output layer(" + outputLayer->name +
                ") should be DetectionOutput, but was " +  outputLayer->type);
        }

        if (outputLayer->params.find("num_classes") == outputLayer->params.end()) {
            throw std::logic_error("Face Detection network output layer (" +
                output + ") should have num_classes integer attribute");
        }
```


6. We check to make sure that the output layers are what we expect them to be.

```cpp
        const int num_classes = outputLayer->GetParamAsInt("num_classes");
        if (labels.size() != num_classes) {
            if (labels.size() == (num_classes - 1))  // if network assumes default "background" class, having no label
                labels.insert(labels.begin(), "fake");
            else
                labels.clear();
        }
```


7. Now, we make sure that the the number of labels we read from the label file match the number of classes in the model.  If not, then we clear the labels and do not use them.  

```cpp
        const InferenceEngine::SizeVector outputDims = _output->dims;
        maxProposalCount = outputDims[1];
        objectSize = outputDims[0];
        if (objectSize != 7) {
            throw std::logic_error("Face Detection network output layer should have 7 as a last dimension");
        }
        if (outputDims.size() != 4) {
            throw std::logic_error("Face Detection network output dimensions not compatible shoulld be 4, but was " + outputDims.size());
        }
```


8. We need to check the output the model is configured to provide and make sure it matches what we are expecting.

```cpp
        _output->setPrecision(Precision::FP32);
        _output->setLayout(Layout::NCHW);
```


9. We configure the output format to use the output precision and memory layout that we expect for results.

```cpp
        slog::info << "Loading Face Detection model to the "<< FLAGS_d << " plugin" << slog::endl;
        input = inputInfo.begin()->first;
        return netReader.getNetwork();
    }
```


10. We save the name of the input blob (inputInfo.begin()->first) for later use when getting a blob for input data.  Finally, we return the InferenceEngine::CNNNetwork object that references this model’s IR.

##### fetchResults()

The last function we define for FaceDetectionClass is fetchResults().

```cpp
    void fetchResults() {
        if (!enabled()) return;
        results.clear();
        if (resultsFetched) return;
        resultsFetched = true;
```


1. First, we make sure that the model is enabled.  If so, we clear out any previous results and track whether results have been fetched which is set to false by submitRequest() to indicate results have not been fetched yet for a request.

```cpp
        const float *detections = request->GetBlob(output)->buffer().as<float *>();
```


2. We get a pointer to the inference model results held in the output blob. 

```cpp
        for (int i = 0; i < maxProposalCount; i++) {
```


3. Now we are going to start looping through the results we got from the face detection model.  "maxProposalCount" has been set to the number of results, or faces, that the model has returned.  

```cpp
            float image_id = detections[i * objectSize + 0];
            Result r;
            r.label = static_cast<int>(detections[i * objectSize + 1]);
            r.confidence = detections[i * objectSize + 2];
            if (r.confidence <= FLAGS_t) {
                continue;
            }

            r.location.x = detections[i * objectSize + 3] * width;
            r.location.y = detections[i * objectSize + 4] * height;
            r.location.width = detections[i * objectSize + 5] * width - r.location.x;
            r.location.height = detections[i * objectSize + 6] * height - r.location.y;
```


4. Here, we retrieve the results from the output blob buffer.  The output format is determined by the model.  For this sample we are expecting to get the following:

    1. Image_id

    2. Label

    3. Confidence 

    4. X coordinate of ROI

    5. Y coordinate of ROI

    6. Width of ROI

```cpp
            if (image_id < 0) {
                break;
            }
```


5. If the returned image_id is not valid, we have run out of data returned from the model, so we exit the loop.

```cpp
            if (FLAGS_r) {
                std::cout << "[" << i << "," << r.label << "] element, prob = " << r.confidence <<
                          "    (" << r.location.x << "," << r.location.y << ")-(" << r.location.width << ","
                          << r.location.height << ")"
                          << ((r.confidence > FLAGS_t) ? " WILL BE RENDERED!" : "") << std::endl;
            }
```


6. Then we check to see if the application was requested to display the confidence information from the model’s inference, and print it to the console if necessary.

```cpp
            results.push_back(r);
        }
    }
```


7. And finally, we take the populated Result object and add it to the vector of results, so we can use it later.

# Using FaceDetectionClass

Now that we have seen what happens behind the scenes in the FaceDetectionClass, we will move into the application code and see how we use it.

See <link to "Integrating Inference Engine in Your Application (Current API)"> for more information on the steps when using a model with the Inference Engine API..

1. Open up an Xterm window or use an existing window to get to a command shell prompt.

2. Change to the directory containing Tutorial Step 2:

```bash
cd tutorials/face_detection_tutorial/step_2
```


3. Open the files "main.cpp" and “face_detection.hpp” in the editor of your choice such as ‘gedit’, ‘gvim’, or ‘vim’.

4. The first set of new header files we need to include are necessary to enable the Inference Engine so that we can load models and use them to analyze the images.

```cpp
#include <inference_engine.hpp>
```


5. There are three headers that let us work with the face detection model and the data it gives us.

```cpp
#include "face_detection.hpp"
#include "mkldnn/mkldnn_extension_ptr.hpp"
#include <ext_list.hpp>
using namespace InferenceEngine;
```


6. Next, we move into the main() function, where we have a few lines that help make it easier to reference plugins for the Inference Engine. The code pairs the input models with their corresponding devices using the command line arguments specifying model and device (this was also covered previously while explaining the path from command line to passing which device to use throught he Inference Engine API).  Then we instantiate our FaceDetection object of type FaceDetectionClass.

```cpp
std::map<std::string, InferencePlugin> pluginsForDevices;
std::vector<std::pair<std::string, std::string>> cmdOptions = {
   {FLAGS_d, FLAGS_m}
};

FaceDetectionClass FaceDetection;
```


7. Now it is time to run through the device/model pairs to create the appropriate plugin for the device that we want to perform inference for the image.  

    1. Loop through pairs pulling device and model arguments to create a new plugin for the specified device if one does not already exist.
    ```cpp
    for (auto && option : cmdOptions) {
       auto deviceName = option.first;
       auto networkName = option.second;
       if (deviceName == "" || networkName == "") {
          continue;
       }
       if (pluginsForDevices.find(deviceName) != pluginsForDevices.end()) {
          continue;
       }
    ```

    2. Load the plugin for the given deviceName then report its version.

    ```cpp
    slog::info << "Loading plugin " << deviceName << slog::endl;
    InferencePlugin plugin = PluginDispatcher({"../../../lib/intel64", ""}).getPluginByDevice(deviceName);
    std::cout << plugin.GetVersion() << std::endl << std::endl;
    ```

    3. If we are loading the CPU plugin, load the available CPU extensions library.

    ```cpp
       /** Load extensions for the CPU plugin **/
       if ((deviceName.find("CPU") != std::string::npos)) {
           plugin.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());
       }
    ```
    

    4. Finish by saving the plugin into the map pluginsForDevices to be used later when loading the model.

      ```cpp
         pluginsForDevices[deviceName] = plugin;
      }
      ```
      

8. Once we have verified that we have the proper plugin, it is time to load the model into the Inference Engine and associate it with the device using the Load helper class previously covered.

```cpp
Load(FaceDetection).into(pluginsForDevices[FLAGS_d]);
```

9. We create some variables that we will use to calculate and report the performance of our application as it processes each image and displays the results.

```cpp
typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
auto wallclock = std::chrono::high_resolution_clock::now();
double ocv_decode_time = 0, ocv_render_time = 0;
```


10. With that done, it is time to enter the application’s "while(true)" main loop to look at what needs to be done to process each image.  We have not included all of the code from Tutorial Step 1, so keep in mind that each time through the loop, the application is grabbing an image and storing it in an object named “frame”.  The code below tracks how long it takes to prepare and queue the current image for the face detection model.

```cpp
auto t0 = std::chrono::high_resolution_clock::now();
FaceDetection.enqueue(frame);
auto t1 = std::chrono::high_resolution_clock::now();
ocv_decode_time = std::chrono::duration_cast<ms>(t1 - t0).count();
```


11. Next, we call the face detection model to infer the image using submitRequest() then we we wait for the results using wait().  The submit-then-wait are enveloped with timing functions to measure how long inference takes.

```cpp
t0 = std::chrono::high_resolution_clock::now();
FaceDetection.submitRequest();
FaceDetection.wait();
t1 = std::chrono::high_resolution_clock::now();
ms detection = std::chrono::duration_cast<ms>(t1 - t0);
```


12. Now that results are ready, we can fetch the results.  We do not need to supply where the results are going to be stored because they are automatically stored in the Result struct that is part of the FaceDetectionClass.

```cpp
FaceDetection.fetchResults();
```


13. The following section of code computes the statistics of how long each step of the analysis process took and prints that information over the image.

```cpp
std::ostringstream out;
out << "OpenCV cap/render time: " << std::fixed << std::setprecision(2)
   << (ocv_decode_time + ocv_render_time) << " ms";
cv::putText(frame, out.str(), cv::Point2f(0, 25), cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(255, 0, 0));
out.str("");
out << "Face detection time  : " << std::fixed << std::setprecision(2) << detection.count()
   << " ms ("
   << 1000.f / detection.count() << " fps)";
cv::putText(frame, out.str(), cv::Point2f(0, 45), cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(255, 0, 0));
```


14. The next set of code displays the more interesting results.  Using OpenCV library calls, we get to see a rectangle drawn around the area that the face detection model determined to be a face.  

```cpp
int i = 0;
for (auto & result : FaceDetection.results) {
   cv::Rect rect = result.location;
   out.str("");
   {
      out << (result.label < FaceDetection.labels.size() ? FaceDetection.labels[result.label] :
         std::string("label #") + std::to_string(result.label))
         << ": " << std::fixed << std::setprecision(3) << result.confidence;
   }
   cv::putText(frame,
                      out.str(),
                      cv::Point2f(result.location.x, result.location.y - 15),
                      cv::FONT_HERSHEY_COMPLEX_SMALL,
                      0.8,
                      cv::Scalar(0, 0, 255));
   auto genderColor =
      cv::Scalar(100, 100, 100);
   cv::rectangle(frame, result.location, genderColor, 1);
   i++;
}
```


15. At this point, we are at the bottom of the "while(true)" loop and all the image inference has been completed.  All that is left to do is output the final results for the frame while measuring the time it took to show the image.

```cpp
t0 = std::chrono::high_resolution_clock::now();
cv::imshow("Detection results", frame);
t1 = std::chrono::high_resolution_clock::now();
ocv_render_time = std::chrono::duration_cast<ms>(t1 - t0).count();
```


16. After all the images have been processed (or you have chosen to stop analyzing input from the camera), if the "-pc" command line argument was used, we print out the final performance statistics to the command window and exit the application.

```cpp
if (FLAGS_pc) {
   FaceDetection.printPerformanceCounts();
}
```


# Building and Running

Now that we have walked through the code and learned what it will do, it is time to build the application and see it in action.

1. Open up an Xterm window or use an existing window to get to a command shell prompt.

2. Change to the directory containing Tutorial Step 2:

```bash
cd tutorials/face_detection_tutorial/step_2
```


3. The first step is to configure the build environment for the Intel OpenVINO toolkit by sourcing the "setupvars.sh" script.

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


6. You now have an executable file to run.  In order to have it run the face detection model, we will need to add a couple of parameters to the command line:

    1. "-i <input-image-or-video-file>" to specify an input image or video file instead of using the USB camera by default

    2. "-m <model-xml-file>"  to specify where to find the module.  For example: -m  /opt/intel/computer_vision_sdk/deployment_tools/intel_models/face-detection-adas-0001/FP32/face-detection-adas-0001.xml”

    3. That is a lot to type and keep straight, so to help make the model names shorter to type  and easier to read, let us use the helper script scripts/setupenv.sh that sets up shell variables we can use.  For reference, here are the contents of scripts/setupenv.sh:
    ```bash
    # Create variables for all models used by the tutorials to make
    
    # check for variable set by setupvars.sh in the SDK, need it to find models
    : ${InferenceEngine_DIR:?Must source the setupvars.sh in the SDK to set InferenceEngine_DIR}
    
    modelDir=$InferenceEngine_DIR/../../intel_models
    
    # Face Detection Model - ADAS
    modName=face-detection-adas-0001
    export mFDA16=$modelDir/$modName/FP16/$modName.xml
    export mFDA32=$modelDir/$modName/FP32/$modName.xml
    
    # Face Detection Model - Retail
    modName=face-detection-retail-0004
    export mFDR16=$modelDir/$modName/FP16/$modName.xml
    export mFDR32=$modelDir/$modName/FP32/$modName.xml
    
    # Age and Gender Model
    modName=age-gender-recognition-retail-0013
    export mAG16=$modelDir/$modName/FP16/$modName.xml
    export mAG32=$modelDir/$modName/FP32/$modName.xml
    
    # Head Pose Estimation Model
    modName=head-pose-estimation-adas-0001
    export mHP16=$modelDir/$modName/FP16/$modName.xml
    export mHP32=$modelDir/$modName/FP32/$modName.xml
    ```
    
    4. To use the script we source it: 

    ```bash
    source ../../scripts/setupenv.sh 
    ```
    
    5. And now we can start referencing the variables for each model as: $mFDA16, $mFDA32, $mFDR16, $mFDR32, $mAG16, $mAG32, $mHP16, $mHP32

7. Again, we will be using images and video files that are included with this tutorial or part of the Intel OpenVINO installation in our sample instructions.  Once you have seen the application working, feel free to try it on your own images and videos.

8. Let us first run it on a single image, to see how it works.

```bash
./intel64/Release/face_detection_tutorial -m $mFDA32 -i ../../data/face.jpg
```


9. You will now see an output window open up with the image displayed.  Over the image, you will see some text with the statistics of how long it took to perform the OpenCV input and output and model processing time.  You should also see a rectangle drawn around the face in the image that has been labeled with an instance number and confidence value.

10. Let us see how our application handles a video file.  And let us also see how easy it is to have our application run a different face detection model by loading the face-detection-retail-0004 IR files by just changing the -m parameter from $mFDA32 to $mFDR32.

```bash
./intel64/Release/face_detection_tutorial -m $mFDR32 -i /opt/intel/computer_vision_sdk/openvx/samples/samples/face_detection/face.mp4
```


11. Now, you should see a window open, playing the video.  Over each frame of the video, you will see a rectangle drawn around the face.  As the face moves around the image, the rectangle will follow it.

12. Finally, let us see how the application works with the default camera input.

```bash
./intel64/Release/face_detection_tutorial -m $mFDA32 -i cam
```


Or

```bash
./intel64/Release/face_detection_tutorial -m $mFDA32
```


13. Now you will see a window displaying the input from the USB camera.  The performance statistics appear over the image, as our application processes each frame.  If there is a face in the image, you will see a rectangle surrounding the face with label and confidence value.  The rectangle will follow the face around the image as it moves and will change sizes to fit the face.

14. When you want to exit the program, make sure the output window is active and press a key.  The output window will close and control will return to the XTerm window.

15. You may remember the final step of the code walk-through, where we showed a step that printed out Performance Counts for the application.  This only happens if you specify the "-pc" command line option.  Let us run the application one more time, to see what kind of output we get from printing those counts.

```bash
./intel64/Release/face_detection_tutorial -m $mFDA32 -i cam -pc
```


16. When you exit the application this time, you will notice that the Xterm window is updated with the final statistics from the face detection model.  The extra output shows a detailed list of the various analysis steps the model performed and how long the model spent on each step.  At the bottom of the list, it displays the total time spent performing the face inference.

```bash
InferenceEngine: 
	API version ............ 1.0
	Build .................. 10073
[ INFO ] Parsing input parameters
[ INFO ] Reading input
[ INFO ] Loading plugin CPU

	API version ............ 1.0
	Build .................. lnx_20180314
	Description ....... MKLDNNPlugin

[ INFO ] Loading network files for Face Detection
[ INFO ] Batch size is set to  1
[ INFO ] Checking Face Detection inputs
[ INFO ] Checking Face Detection outputs
[ INFO ] Loading Face Detection model to the CPU plugin
[ INFO ] Start inference 
[ INFO ] Press 's' key to save a snapshot, press any other key to stop
[ INFO ] Press 's' key to save a snapshot, press any other key to exit
[ INFO ] Performance counts for Face Detection

Mul_788/Fused_Mul_1070/Fus... EXECUTED       layerType: ScaleShift         realTime: 946        cpu: 946            execType: unknown
conv1                         EXECUTED       layerType: Convolution        realTime: 2586       cpu: 2586           execType: jit_sse42
conv2_1/dw                    EXECUTED       layerType: Convolution        realTime: 1484       cpu: 1484           execType: jit_sse42_dw
conv2_1/sep                   EXECUTED       layerType: Convolution        realTime: 5594       cpu: 5594           execType: jit_sse42_1x1
conv2_2/dw                    EXECUTED       layerType: Convolution        realTime: 1860       cpu: 1860           execType: jit_sse42_dw
Total time: 91233    microseconds
[ INFO ] Execution successful
```


17. Above, is part of the output you will see in your console window.  It shows information on what the Inference Engine loaded, followed by the performance statistics gathered from running each layer within the model.  This includes the calculation run, the model layer type, the real time spent performing the calculation, the CPU time spent performing the calculation, and the type of calculation that was performed.  In this instance, since we loaded the model onto the CPU, the "realTime" and “cpu” time values are the same.  The last bit of information we see is the total time spent spent performing the face analysis.  In this example running on an UP Squared Apollo Lake Intel Pentium N4200, it was 91233 microseconds, or 0.091233 seconds.

# Conclusion

Congratulations on using a CNN model to detect faces!  You have now seen that the process can be done quite quickly.  The classes and helper functions that we added here are aimed at making it easy to add more models to our application by following the same pattern.  We will see it again in action in Step 3, when we add an age and gender inferring model, and then again in Step 4, when we add head pose estimation.

# Navigation
[Face Detection Tutorial](../Readme.md)

[Face Detection Tutorial Step 1](../step_1/Readme.md)

[Face Detection Tutorial Step 3](../step_3/Readme.md)