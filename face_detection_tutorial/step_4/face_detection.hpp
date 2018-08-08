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
#include <gflags/gflags.h>
#include <iostream>

#ifdef _WIN32
#include <os/windows/w_dirent.h>
#else
#include <dirent.h>
#endif

/// @brief message for help argument
static const char help_message[] = "Print a usage message.";

/// @brief message for images argument
static const char video_message[] = "Optional. Path to an video file. Default value is \"cam\" to work with camera.";

/// @brief message for model argument
static const char face_detection_model_message[] = "Required. Path to an .xml file with a trained face detection model.";
static const char age_gender_model_message[] = "Optional. Path to an .xml file with a trained age gender model.";
static const char head_pose_model_message[] = "Optional. Path to an .xml file with a trained head pose model.";

/// @brief message for plugin argument
static const char plugin_message[] = "Plugin name. For example MKLDNNPlugin. If this parameter is pointed, " \
"the sample will look for this plugin only.";

/// @brief message for assigning face detection calculation to device
static const char target_device_message[] = "Specify the target device for Face Detection (CPU, GPU, FPGA, or MYRIAD. " \
"Sample will look for a suitable plugin for device specified.";

/// @brief message for assigning age gender calculation to device
static const char target_device_message_ag[] = "Specify the target device for Age Gender Detection (CPU, GPU, FPGA, or MYRIAD. " \
"Sample will look for a suitable plugin for device specified.";

/// @brief message for number of simultaneously age gender detections using dynamic batch
static const char num_batch_ag_message[] = "Specify number of maximum simultaneously processed faces for Age Gender Detection ( default is 1).";

/// @brief message for assigning age gender calculation to device
static const char target_device_message_hp[] = "Specify the target device for Head Pose Detection (CPU, GPU, FPGA, or MYRIAD. " \
"Sample will look for a suitable plugin for device specified.";

/// @brief message for number of simultaneously age gender detections using dynamic batch
static const char num_batch_hp_message[] = "Specify number of maximum simultaneously processed faces for Head Pose Detection ( default is 1).";

/// @brief message for performance counters
static const char performance_counter_message[] = "Enables per-layer performance report.";

/// @brief message for clDNN custom kernels desc
static const char custom_cldnn_message[] = "Required for clDNN (GPU)-targeted custom kernels."\
"Absolute path to the xml file with the kernels desc.";

/// @brief message for user library argument
static const char custom_cpu_library_message[] = "Required for MKLDNN (CPU)-targeted custom layers." \
"Absolute path to a shared library with the kernels impl.";

/// @brief message for probability threshold argument
static const char thresh_output_message[] = "Probability threshold for detections.";

/// @brief message raw output flag
static const char raw_output_message[] = "Inference results as raw values.";

/// @brief message no wait for keypress after input stream completed
static const char no_wait_for_keypress_message[] = "No wait for key press in the end.";

/// @brief message no show processed video
static const char no_show_processed_video[] = "No show processed video.";


/// \brief Define flag for showing help message <br>
DEFINE_bool(h, false, help_message);

/// \brief Define parameter for set image file <br>
/// It is a required parameter
DEFINE_string(i, "cam", video_message);

/// \brief Define parameter for face detection  model file <br>
/// It is a required parameter
DEFINE_string(m, "", face_detection_model_message);

/// \brief Define parameter for face detection  model file <br>
/// It is a required parameter
DEFINE_string(m_ag, "", age_gender_model_message);

/// \brief Define parameter for face detection  model file <br>
/// It is a required parameter
DEFINE_string(m_hp, "", head_pose_model_message);

/// \brief device the target device for face detection infer on <br>
DEFINE_string(d, "CPU", target_device_message);

/// \brief device the target device for age gender detection on <br>
DEFINE_string(d_ag, "CPU", target_device_message_ag);

/// \brief device the target device for age gender detection on <br>
DEFINE_uint32(n_ag, 1, num_batch_ag_message);

/// \brief device the target device for head pose detection on <br>
DEFINE_string(d_hp, "CPU", target_device_message_hp);

/// \brief device the target device for head pose detection on <br>
DEFINE_uint32(n_hp, 1, num_batch_hp_message);

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

/**
* \brief This function show a help message
*/
static void showUsage() {
    std::cout << std::endl;
    std::cout << "face_detection_tutorial [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                         " << help_message << std::endl;
    std::cout << "    -i \"<path>\"                " << video_message << std::endl;
    std::cout << "    -m \"<path>\"                " << face_detection_model_message<< std::endl;
    std::cout << "    -m_ag \"<path>\"             " << age_gender_model_message << std::endl;
    std::cout << "    -m_hp \"<path>\"             " << head_pose_model_message << std::endl;
    std::cout << "      -l \"<absolute_path>\"     " << custom_cpu_library_message << std::endl;
    std::cout << "          Or" << std::endl;
    std::cout << "      -c \"<absolute_path>\"     " << custom_cldnn_message << std::endl;
    std::cout << "    -d \"<device>\"              " << target_device_message << std::endl;
    std::cout << "    -d_ag \"<device>\"           " << target_device_message_ag << std::endl;
    std::cout << "    -d_hp \"<device>\"           " << target_device_message_hp << std::endl;
    std::cout << "    -n_ag \"<num>\"              " << num_batch_ag_message << std::endl;
    std::cout << "    -n_hp \"<num>\"              " << num_batch_hp_message << std::endl;
    std::cout << "    -no_wait                   " << no_wait_for_keypress_message << std::endl;
    std::cout << "    -no_show                   " << no_show_processed_video << std::endl;
    std::cout << "    -pc                        " << performance_counter_message << std::endl;
    std::cout << "    -r                         " << raw_output_message << std::endl;
    std::cout << "    -t                         " << thresh_output_message << std::endl;
}
