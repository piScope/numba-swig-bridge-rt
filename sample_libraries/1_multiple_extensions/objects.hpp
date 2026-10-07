#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace nsb_sample1 {

class Buffer {
public:
    explicit Buffer(int size);
    [[nodiscard]] int Size() const noexcept;
    [[nodiscard]] double Get(int index) const;
    void Set(int index, double value);
    [[nodiscard]] double Sum() const noexcept;
    // Returns a borrowed pointer to this object, or nullptr.  It exercises
    // nullable selected-object returns without transferring ownership.
    [[nodiscard]] Buffer* MaybeSelf(bool present) noexcept {
        return present ? this : nullptr;
    }
private:
    std::vector<double> values_;
};

class Counter {
public:
    explicit Counter(std::int64_t initial = 0) noexcept;
    void Add(std::int64_t amount) noexcept;
    [[nodiscard]] std::int64_t Value() const noexcept;
    [[nodiscard]] bool AtLeast(std::int64_t value) const noexcept;
private:
    std::int64_t value_;
};

}  // namespace nsb_sample1
