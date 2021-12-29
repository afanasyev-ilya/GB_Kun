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
    explicit Descriptor() : debug_flag(false)
    {
        for(auto & i : desc_)
            i = GrB_DEFAULT;
    }

    ~Descriptor() = default;

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
    Desc_value desc_[GrB_NDESCFIELD]{};
    bool debug_flag;
};

}
}