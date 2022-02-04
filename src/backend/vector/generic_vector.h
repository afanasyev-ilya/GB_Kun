#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
class GenericVector
{
public:

public:
    virtual VNT get_nvals() const = 0;
    virtual void print_storage_type() const = 0;
    virtual void print() const = 0;
    virtual void set_element(T _val, VNT _pos) = 0;
    virtual void set_all_constant(T _val) = 0;
    virtual void fill_with_zeros() = 0;
    virtual VNT get_size() const = 0;
    virtual LA_Info fillAscending(Index nvals) = 0;
    virtual void dup (GenericVector<T>* rhs) = 0;
    virtual bool isDense() const = 0;
    virtual bool isSparse() const = 0;
    virtual T* get_vals() const = 0;
    virtual VNT* get_ids() const = 0;


};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
