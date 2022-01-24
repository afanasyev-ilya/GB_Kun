#pragma once

#include "../backend/descriptor/descriptor.h"
namespace lablas {

typedef pair<Desc_field, Desc_value> desc_pair;

class Descriptor {
public:
    Descriptor(): _descriptor() {}

    explicit Descriptor(const vector<desc_pair>& _init_data): _descriptor()
    {
        for(auto i : _init_data)
        {
            _descriptor.set(i.first, i.second);
        }
    }

    ~Descriptor()= default;

    LA_Info set(Desc_field field, Desc_value value)
    {
        return _descriptor.set(field, value);
    }

    LA_Info get(Desc_field field, Desc_value* value)
    {
        return _descriptor.get(field, value);
    }

    void toggle(Desc_field field)
    {
         // TODO
    }

    Desc_value get(Desc_field field)
    {
        Desc_value value;
        _descriptor.get(field, &value);
        return value;
    }

    backend::Descriptor* get_descriptor()
    {
        return &_descriptor;
    }

private:
    backend::Descriptor _descriptor;
};

lablas::Descriptor GrB_NULL;
lablas::Descriptor GrB_DESC_RSC({{GrB_OUTPUT, GrB_REPLACE}, {GrB_MASK, GrB_STR_COMP}});
lablas::Descriptor GrB_DESC_C({{GrB_MASK, GrB_COMP}});
lablas::Descriptor GrB_DESC_S({{GrB_MASK, GrB_STRUCTURE}});
lablas::Descriptor GrB_DESC_SC({{GrB_MASK, GrB_STR_COMP}});

}
