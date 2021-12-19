#pragma once

#include "../backend/descriptor/descriptor.h"
namespace lablas {

class Descriptor {
public:
    Descriptor(): _descriptor() {}
    ~Descriptor(){}

    LA_Info set(Desc_field field, Desc_value value) {
        return _descriptor.set(field, value);
    }
    LA_Info get(Desc_field field, Desc_value* value) {
        return _descriptor.get(field, value);
    }
    backend::Descriptor* get_descriptor() {
        return &_descriptor;
    }

private:
    backend::Descriptor _descriptor;
};

}
