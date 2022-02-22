#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
class GenericVector
{
protected:
    string name;
public:
    virtual VNT get_nvals() const = 0;
    virtual VNT get_size() const = 0;

    virtual void print_storage_type() const = 0;
    virtual void print() const = 0;
    virtual void set_element(T _val, VNT _pos) = 0;
    virtual void set_all_constant(T _val) = 0;
    virtual void fill_with_zeros() = 0;

    virtual Storage get_storage() = 0;

    void set_name(const string &_name) {name = _name;};
    const string &get_name() { return name; };
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
