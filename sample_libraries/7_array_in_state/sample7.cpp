#include "directors.hpp"

namespace nsb_sample7 {

double invoke_accumulate(const Accumulator& accumulator, double value) {
    return accumulator.Accumulate(value);
}

} // namespace nsb_sample7
