#pragma once

#include "../../cpp_graphblas/types.hpp"
#include "../la_backend.h"
#include "../../helpers/memory_API/memory_API.h"

#define _matrix_size 100

namespace lablas{
namespace backend{

class Descriptor
{
public:
    explicit Descriptor() : desc_{ GrB_DEFAULT, GrB_DEFAULT, GrB_DEFAULT, GrB_DEFAULT,
                                          GrB_FIXEDROW, GrB_32, GrB_32, GrB_128, GrB_PUSHPULL,
                                          GrB_16, GrB_CUDA}, debug_flag(false)
    {

    }
    ~Descriptor()
    {

    }

    LA_Info set(Desc_field field, Desc_value value) {
        desc_[field] = value;
        return GrB_SUCCESS;
    }

    LA_Info get(Desc_field field, Desc_value *value) {
        *value = desc_[field];
        return GrB_SUCCESS;
    }

    bool debug() const{
        return debug_flag;
    }

private:
    Desc_value desc_[GrB_NDESCFIELD];
    bool debug_flag;
};

}
}