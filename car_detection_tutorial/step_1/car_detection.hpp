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


/// @brief message for help argument
static const char help_message[] = "Print a usage message";

/// @brief message for images argument
static const char video_message[] = "Path to a video file or \"cam\" to work with camera";


/// @brief message no wait for keypress after input stream completed
static const char no_wait_for_keypress_message[] = "No wait for key press in the end";

/// @brief message no show processed video
static const char no_show_processed_video[] = "No show processed video";

/// \brief Define flag for showing help message <br>
DEFINE_bool(h, false, help_message);

/// \brief Define parameter for set image file <br>
/// It is a required parameter
DEFINE_string(i, "cam", video_message);


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
    printParameter("no_wait");
    printParameter("no_show");
}
