#pragma once

namespace nsb_sample3 {
class Base {
public:
    explicit Base(int value);
    virtual ~Base() = default; // Required for checked dynamic downcasts.
    int BaseMethod() const;
    virtual int VirtualMethod() const { return BaseMethod(); }
private:
    int value_;
};

// BaseMethod is inherited; no forwarding method or redefinition is provided.
class Derived : public Base {
public:
    explicit Derived(int value);
    int DerivedMethod() const;
    const Base &AsBase() const { return *this; }
    int VirtualMethod() const override { return DerivedMethod(); }
};

class GrandDerived : public Derived {
public:
    explicit GrandDerived(int value) : Derived(value) {}
};

class Side {
public:
    virtual ~Side() = default;
    int SideMethod() const { return 100; }
private:
    int padding_ = 100;
};

// Base is reached through the second direct base, requiring pointer adjustment.
class Multiple : public Side, public Derived {
public:
    explicit Multiple(int value) : Derived(value) {}
    bool BaseIsOffset() const {
        return static_cast<const void*>(static_cast<const Base*>(this)) !=
               static_cast<const void*>(this);
    }
};

class VirtualDerived : public virtual Base {
public:
    explicit VirtualDerived(int value) : Base(value) {}
};

class Alternate : public Base {
public:
    explicit Alternate(int value) : Base(value) {}
};

// Two Base subobjects: upcast(obj, Base) must be rejected as ambiguous.
class Ambiguous : public Derived, public Alternate {
public:
    explicit Ambiguous(int value) : Derived(value), Alternate(value + 1) {}
};

class Hiding : public Derived {
public:
    explicit Hiding(int value) : Derived(value) {}
    int BaseMethod(int extra) const { return Base::BaseMethod() + extra; }
};

class Consumer {
public:
    int Read(const Base& value) const { return value.BaseMethod(); }
};

class Plain {
public:
    int PlainMethod() const { return 1; }
};
class PlainDerived : public Plain {};

class Private : private Base {
public:
    explicit Private(int value) : Base(value) {}
    int OwnMethod() const { return BaseMethod(); }
};
} // namespace nsb_sample3
