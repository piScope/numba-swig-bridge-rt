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

class GradientBase {
public:
    virtual ~GradientBase();
    virtual int Value() const = 0;
};

class Gradient : public GradientBase {
public:
    explicit Gradient(int value) : value_(value) { }
    int Value() const override { return value_; }
    void SetValue(int value) { value_ = value; }
private:
    int value_;
};

class Form {
public:
    Form() = default;
    ~Form();
    GradientBase& GetGradient(int value);
private:
    Gradient* gradient_ = nullptr;
};

class GradientConsumer {
public:
    GradientConsumer() = default;
    virtual ~GradientConsumer() = default;
    virtual void SetOperator(const GradientBase& gradient) = 0;
    virtual int Read() const = 0;
};

int ConsumeGradient(GradientConsumer& consumer, Form& form, int value);
int DestroyedGradientCount();

int ConsumeReference(Provider &provider);
int ConsumePointer(Provider &provider, bool present);
int DestroyedParentCount();

}  // namespace nsb_sample9
