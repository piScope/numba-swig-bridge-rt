#pragma once

namespace nsb_sample7 {

class Accumulator {
public:
    Accumulator() = default;
    virtual ~Accumulator() = default;
    virtual double Accumulate(double value) const = 0;
};

double invoke_accumulate(const Accumulator& accumulator, double value);

} // namespace nsb_sample7
