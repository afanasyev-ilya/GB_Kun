#pragma once

namespace lablas {
namespace backend {

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class VertexGroup
{
public:
    ~VertexGroup()
    {
        MemoryAPI::free_array(opt_data);
        opt_data = NULL;
    }

    bool in_range(ENT _connections_num) const
    {
        if(_connections_num >= min_threshold && _connections_num < max_threshold)
            return true;
        else
            return false;
    }

    void set_thresholds(ENT _min_threshold, ENT _max_threshold)
    {
        min_threshold = _min_threshold;
        max_threshold = _max_threshold;
        data.clear();
    }

    void push_back(VNT _row)
    {
        data.push_back(_row);
    }

    VNT get_size() const
    {
        return size;
    }

    const VNT *get_data() const
    {
        return opt_data;
    }

    void finalize_creation(int _target_socket)
    {
        size = (VNT)data.size();
        MemoryAPI::numa_aware_alloc(&opt_data, size, _target_socket);
        MemoryAPI::copy(opt_data, &data[0], size);
    }

    void deep_copy(VertexGroup &_copy, int _target_socket)
    {
        this->size = _copy.size;
        this->min_threshold = _copy.size;
        this->max_threshold = _copy.size;

        MemoryAPI::numa_aware_alloc(&(this->opt_data), _copy.size, _target_socket);
        MemoryAPI::copy(this->opt_data, _copy.opt_data, _copy.size);
    }

    void replace_data(VNT *_new_data)
    {
        MemoryAPI::free_array(opt_data);
        opt_data = _new_data;
    }
private:
    ENT min_threshold;
    ENT max_threshold;

    std::vector<VNT> data;
    VNT *opt_data;
    VNT size;
};

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
