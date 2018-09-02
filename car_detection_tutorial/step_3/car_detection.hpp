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

static const char mVLP16[] = MOD2PATH("vehicle-license-plate-detection-barrier-0106", FP16);
static const char mVLP32[] = MOD2PATH("vehicle-license-plate-detection-barrier-0106", FP32);

static const char mVDR16[] = MOD2PATH("vehicle-detection-adas-0002", FP16);
static const char mVDR32[] = MOD2PATH("vehicle-detection-adas-0002", FP32);

static const char mVA16[] = MOD2PATH("vehicle-attributes-recognition-barrier-0039", FP16);
static const char mVA32[] = MOD2PATH("vehicle-attributes-recognition-barrier-0039", FP32);

// make each path a parameter that can be referenced as "$*" when setting other parameters.
//    Example: m=$mVLP16 will result in PARAMETERS_m=value-of(mVLP16)
//    As parameters they can also be overwritten if desired.
static const char model_path_message[] = "Path to model";
DEFINE_string(mVLP32, mVLP32, model_path_message);
DEFINE_string(mVLP16, mVLP16, model_path_message);
DEFINE_string(mVDR32, mVDR32, model_path_message);
DEFINE_string(mVDR16, mVDR16, model_path_message);
DEFINE_string(mVA32, mVA32, model_path_message);
DEFINE_string(mVA16, mVA16, model_path_message);


/// @brief message for help argument
static const char help_message[] = "Print a usage message";

/// @brief message for images argument
static const char video_message[] = "Path to a video file or \"cam\" to work with camera";

/// @brief message for model argument
static const char vehicle_detection_model_message[] = "Path to the Vehicle/License-Plate Detection model (.xml) file";
static const char vehicle_attribs_model_message[] = "Path to the Vehicle Attributes model (.xml) file";

/// @brief message for assigning vehicle detection inference to device
static const char target_device_message[] = "Specify the target device for Vehicle Detection (CPU, GPU, FPGA, or MYRIAD)";

/// @brief message for number of simultaneously vehicle detections using dynamic batch
static const char num_batch_message[] = "Specify number of maximum simultaneously processed frames for Vehicle Detection";

/// @brief message for assigning vehicle attributes to device
static const char target_device_message_vehicle_attribs[] = "Specify the target device for Vehicle Attributes (CPU, GPU, FPGA, or MYRIAD)";

/// @brief message for number of simultaneously vehicle attributes detections using dynamic batch
static const char num_batch_va_message[] = "Specify number of maximum simultaneously processed vehicles for Vehicle Attributes Detection ( default is 1)";

/// @brief message for enabling dynamic batching for vehicle detections
static const char dyn_va_message[] = "Enable dynamic batching for Vehicle Attributes Detection ( default is 0).";

/// @brief message auto_resize input flag
static const char auto_resize_message[] = "Enable auto-resize (ROI crop & data resize) of input during inference.";

/// @brief message for performance counters
static const char performance_counter_message[] = "Enable per-layer performance report.";

/// @brief message for clDNN custom kernels desc
static const char custom_cldnn_message[] = "Required for clDNN (GPU)-targeted custom kernels."\
"Absolute path to the xml file with the kernels desc";

/// @brief message for user library argument
static const char custom_cpu_library_message[] = "Required for MKLDNN (CPU)-targeted custom layers." \
"Absolute path to a shared library with the kernels impl";

/// @brief message for probability threshold argument
static const char thresh_output_message[] = "Probability threshold for vehicle/licence-plate detections";

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

/// \brief Define parameter for vehicle detection  model file <br>
/// It is a required parameter
DEFINE_string(m, mVLP32, vehicle_detection_model_message);

/// \brief Define parameter for vehicle attributes model file <br>
/// It is a required parameter
DEFINE_string(m_va, mVA32, vehicle_attribs_model_message);

/// \brief device the target device for vehicle detection infer on <br>
DEFINE_string(d, "GPU", target_device_message);

/// \brief batch size for running vehicle detection <br>
DEFINE_uint32(n, 1, num_batch_message);

/// \brief device the target device for vehicle attributes detection on <br>
DEFINE_string(d_va, "GPU", target_device_message_vehicle_attribs);

/// \brief device the target device for vehicle attributes detection on <br>
DEFINE_uint32(n_va, 1, num_batch_va_message);

/// \brief Define flag for enabling dynamic batching for vehicle attributes detection <br>
DEFINE_bool(dyn_va, false, dyn_va_message);

/// \brief Define flag for enabling auto-resize of inputs for all models <br>
DEFINE_bool(auto_resize, false, auto_resize_message);

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
    std::cout << "car_detection_tutorial current parameter settings:" << std::endl;
    printParameter("i");
    printParameter("m");
    printParameter("d");
    printParameter("n");
    printParameter("m_va");
    printParameter("d_va");
    printParameter("n_va");
    printParameter("dyn_va");
    printParameter("auto_resize");
    printParameter("t");
    printParameter("r");
    printParameter("no_wait");
    printParameter("no_show");
    printParameter("pc");
//    printParameter("l");
//    printParameter("c");
}
