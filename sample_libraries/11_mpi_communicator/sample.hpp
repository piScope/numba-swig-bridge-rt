#pragma once
#ifndef SWIG
#include <mpi.h>
#endif
class CommCallback {
public:
    CommCallback() = default;
    virtual ~CommCallback() = default;
    virtual int Value() const = 0;
};
class CommConsumer {
    MPI_Comm comm;
public:
    explicit CommConsumer(MPI_Comm value) : comm(value) {}
    MPI_Comm GetComm() const { return comm; }
    int Apply(const CommCallback &callback) const { return callback.Value(); }
};
