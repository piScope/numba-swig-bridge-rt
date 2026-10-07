#pragma once

namespace nsb_sample9 {

class Child {
public:
    explicit Child(int value = 0) : value_(value) { }
    int Value() const { return value_; }
    void SetValue(int value) { value_ = value; }

private:
    int value_;
};

class Parent {
public:
    explicit Parent(int value = 0) : child_(value) { }
    explicit Parent(int *value) : child_(*value) { }
    ~Parent();
    Child &GetChild() { return child_; }
    const Child &GetChild() const { return child_; }
    Child *MaybeChild(bool present) { return present ? &child_ : nullptr; }

private:
    Child child_;
};

class Provider {
public:
    Provider() = default;
    virtual ~Provider() = default;
    virtual Child &GetChild() = 0;
    virtual Child *MaybeChild(bool present) = 0;
};

int ConsumeReference(Provider &provider);
int ConsumePointer(Provider &provider, bool present);
int DestroyedParentCount();

}  // namespace nsb_sample9
