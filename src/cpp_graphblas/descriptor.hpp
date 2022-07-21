/*
 * This file uses definitions and source code from graphblast, which is available under
 * "Apache License 2.0" license. For details, see https://github.com/gunrock/graphblast/blob/master/LICENSE
 * */
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

    LA_Info toggle(Desc_field field)
    {
         return _descriptor.toggle(field);
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
lablas::Descriptor GrB_DESC_IKJ({{GrB_MXMMODE, GrB_IKJ}});
lablas::Descriptor GrB_DESC_IKJ_MASKED({{GrB_MXMMODE, GrB_IKJ_MASKED}});
lablas::Descriptor GrB_DESC_IJK({{GrB_MXMMODE, GrB_IJK}});
lablas::Descriptor GrB_DESC_IJK_DOUBLE_SORT({{GrB_MXMMODE, GrB_IJK_DOUBLE_SORT}});
lablas::Descriptor GrB_DESC_ESC({{GrB_MXMMODE, GrB_ESC}});
lablas::Descriptor GrB_DESC_ESC_MASKED({{GrB_MXMMODE, GrB_ESC_MASKED}});

}
