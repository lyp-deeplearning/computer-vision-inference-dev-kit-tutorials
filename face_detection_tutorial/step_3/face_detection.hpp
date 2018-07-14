/*
// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <string>
#include <vector>
#include <iostream>

#include "Parameters.hpp"

#ifdef _WIN32
#include <os/windows/w_dirent.h>
#else
#include <dirent.h>
#endif

// models
// Set to top directory where models are installed
#define MODEL_DIR "/opt/intel/computer_vision_sdk/deployment_tools/intel_models/"
#define MOD2PATH(modName, fPrec) MODEL_DIR "/" modName "/" #fPrec "/" modName ".xml"

static const char mFDA16[] = MOD2PATH("face-detection-adas-0001", FP16);
static const char mFDA32[] = MOD2PATH("face-detection-adas-0001", FP32);

static const char mFDR16[] = MOD2PATH("face-detection-retail-0004", FP16);
static const char mFDR32[] = MOD2PATH("face-detection-retail-0004", FP32);

static const char mAG16[] = MOD2PATH("age-gender-recognition-retail-0013", FP16);
static const char mAG32[] = MOD2PATH("age-gender-recognition-retail-0013", FP32);

// make each path a parameter that can be referenced as "$*" when setting other parameters.
//    Example: m=$mFDA16 will result in PARAMETERS_m=value-of(mFDA16)
//    As parameters they can also be overwritten if desired.
static const char model_path_message[] = "Path to model";
DEFINE_string(mFDA32, mFDA32, model_path_message);
DEFINE_string(mFDA16, mFDA16, model_path_message);
DEFINE_string(mFDR32, mFDR32, model_path_message);
DEFINE_string(mFDR16, mFDR16, model_path_message);
DEFINE_string(mAG32, mAG32, model_path_message);
DEFINE_string(mAG16, mAG16, model_path_message);


/// @brief message for help argument
static const char help_message[] = "Print a usage message";

/// @brief message for images argument
static const char video_message[] = "Path to a video file or \"cam\" to work with camera";

/// @brief message for model argument
static const char face_detection_model_message[] = "Path to an .xml file with a trained face detection model";
static const char age_gender_model_message[] = "Path to an .xml file with a trained age gender model";

/// @brief message for plugin argument
static const char plugin_message[] = "Plugin name. For example MKLDNNPlugin. If this parameter is pointed, " \
"the sample will look for this plugin only.";

/// @brief message for assigning face detection calculation to device
static const char target_device_message[] = "Specify the target device for Face Detection (CPU, GPU, FPGA, or MYRIAD)";

/// @brief message for assigning age gender calculation to device
static const char target_device_message_ag[] = "Specify the target device for Age Gender Detection (CPU, GPU, FPGA, or MYRIAD)";

/// @brief message for number of simultaneously age gender detections using dynamic batch
static const char num_batch_ag_message[] = "Specify number of maximum simultaneously processed faces for Age Gender Detection";

/// @brief message for performance counters
static const char performance_counter_message[] = "Enable per-layer performance report.";

/// @brief message for clDNN custom kernels desc
static const char custom_cldnn_message[] = "Required for clDNN (GPU)-targeted custom kernels."\
"Absolute path to the xml file with the kernels desc";

/// @brief message for user library argument
static const char custom_cpu_library_message[] = "Required for MKLDNN (CPU)-targeted custom layers." \
"Absolute path to a shared library with the kernels impl";

/// @brief message for probability threshold argument
static const char thresh_output_message[] = "Probability threshold for detections";

/// @brief message raw output flag
static const char raw_output_message[] = "Print inference results as raw values";

/// @brief message no wait for keypress after input stream completed
static const char no_wait_for_keypress_message[] = "No wait for key press in the end";

/// @brief message no show processed video
static const char no_show_processed_video[] = "No show processed video";


/// \brief Define flag for showing help message <br>
DEFINE_bool(h, false, help_message);

/// \brief Define parameter for set image file <br>
/// It is a required parameter
DEFINE_string(i, "cam", video_message);

/// \brief Define parameter for face detection  model file <br>
/// It is a required parameter
DEFINE_string(m, mFDA32, face_detection_model_message);

/// \brief Define parameter for face detection  model file <br>
/// It is a required parameter
DEFINE_string(m_ag, mAG32, age_gender_model_message);

/// \brief device the target device for face detection infer on <br>
DEFINE_string(d, "GPU", target_device_message);

/// \brief device the target device for age gender detection on <br>
DEFINE_string(d_ag, "GPU", target_device_message_ag);

/// \brief device the target device for age gender detection on <br>
DEFINE_uint32(n_ag, 1, num_batch_ag_message);

/// \brief Enable per-layer performance report
DEFINE_bool(pc, false, performance_counter_message);

/// @brief clDNN custom kernels path <br>
/// Default is ./lib
DEFINE_string(c, "", custom_cldnn_message);

/// @brief Absolute path to CPU library with user layers <br>
/// It is a optional parameter
DEFINE_string(l, "", custom_cpu_library_message);

/// \brief Flag to output raw scoring results<br>
/// It is an optional parameter
DEFINE_bool(r, false, raw_output_message);

/// \brief Flag to output raw scoring results<br>
/// It is an optional parameter
DEFINE_double(t, 0.5, thresh_output_message);

/// \brief Flag to disable keypress exit<br>
/// It is an optional parameter
DEFINE_bool(no_wait, false, no_wait_for_keypress_message);

/// \brief Flag to disable processed video showing<br>
/// It is an optional parameter
DEFINE_bool(no_show, false, no_show_processed_video);

static void printParameter(const char* param_name) {
    parameters::ParameterVal* paramT = parameters::findParameterByName(param_name);
	if (paramT != NULL) {
		std::cout << "- " << paramT->desc;
		std::cout << " (type=" << paramT->type << "):";
		std::cout << std::endl;
		std::cout << "   " << paramT->name << "=";
		paramT->printVal(std::cout);
		std::cout << std::endl;
	}
}

/**
* \brief This function displays current parameter settings
*/
static void showParameters() {
    std::cout << std::endl;
    std::cout << "face_detection_tutorial current parameter settings:" << std::endl;
    printParameter("i");
    printParameter("m");
    printParameter("d");
    printParameter("m_ag");
    printParameter("d_ag");
    printParameter("n_ag");
    printParameter("t");
    printParameter("r");
    printParameter("no_wait");
    printParameter("no_show");
    printParameter("pc");
//    printParameter("l");
//    printParameter("c");
}
