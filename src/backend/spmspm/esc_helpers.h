/// @file esc_helpers.h
/// @author Lastname:Firstname
/// @version Revision 1.1
/// @brief Helpers for ESC SpMSpM algorithm
/// @details Implements Helpers for ESC SpMSpM algorithm such as dynamic tuning parametrization
/// @date June 13, 2022

#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// @namespace Lablas
namespace lablas {

/// @namespace Backend
namespace backend {

/// @namespace ESC_Helpers
namespace esc_helpers {

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// @brief Get nearest power of two for ESC algorithm
///
/// Implements function that returns nearest power of two for the input value that could be used in dynamic parameter
/// tuning for ESC algorithms.
/// @param[in] val Input value
/// result Nearest power of two integer value
unsigned long long get_nearest_power_of_two(unsigned long long val) {
    val--;
    val |= val >> 1;
    val |= val >> 2;
    val |= val >> 4;
    val |= val >> 8;
    val |= val >> 16;
    val |= val >> 32;
    val++;
    return val;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

}
}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
