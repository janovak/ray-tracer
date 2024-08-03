#pragma once

#include "constants.h"

struct Interval {
    float m_min;
    float m_max;

    __device__ Interval() : m_min(+kInfinity), m_max(-kInfinity) {} // Default interval is empty

    __device__ Interval(float min, float max) : m_min(min), m_max(max) {}

    __device__ float Size() const {
        return m_max - m_min;
    }

    __device__ bool Contains(float x) const {
        return m_min <= x && x <= m_max;
    }

    __device__ bool Surrounds(float x) const {
        return m_min < x && x < m_max;
    }

    static const Interval empty, universe;
};

const Interval Interval::empty    = Interval(+kInfinity, -kInfinity);
const Interval Interval::universe = Interval(-kInfinity, +kInfinity);