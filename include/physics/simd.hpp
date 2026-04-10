#pragma once

#include <cstddef>
#include <cstring>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#elif defined(__SSE__)
#include <xmmintrin.h>
#endif

namespace physics::simd {

#if defined(__ARM_NEON)

constexpr bool available = true;
constexpr std::size_t width = 4;
constexpr const char* backend_name = "neon";

struct Vec4f {
    float32x4_t value;
};

inline Vec4f splat(float value) {
    return {vdupq_n_f32(value)};
}

inline Vec4f load_u(const float* source) {
    return {vld1q_f32(source)};
}

inline void store_u(float* destination, Vec4f value) {
    vst1q_f32(destination, value.value);
}

inline Vec4f operator+(Vec4f lhs, Vec4f rhs) {
    return {vaddq_f32(lhs.value, rhs.value)};
}

inline Vec4f operator-(Vec4f lhs, Vec4f rhs) {
    return {vsubq_f32(lhs.value, rhs.value)};
}

inline Vec4f operator*(Vec4f lhs, Vec4f rhs) {
    return {vmulq_f32(lhs.value, rhs.value)};
}

#elif defined(__SSE__)

constexpr bool available = true;
constexpr std::size_t width = 4;
constexpr const char* backend_name = "sse";

struct Vec4f {
    __m128 value;
};

inline Vec4f splat(float value) {
    return {_mm_set1_ps(value)};
}

inline Vec4f load_u(const float* source) {
    return {_mm_loadu_ps(source)};
}

inline void store_u(float* destination, Vec4f value) {
    _mm_storeu_ps(destination, value.value);
}

inline Vec4f operator+(Vec4f lhs, Vec4f rhs) {
    return {_mm_add_ps(lhs.value, rhs.value)};
}

inline Vec4f operator-(Vec4f lhs, Vec4f rhs) {
    return {_mm_sub_ps(lhs.value, rhs.value)};
}

inline Vec4f operator*(Vec4f lhs, Vec4f rhs) {
    return {_mm_mul_ps(lhs.value, rhs.value)};
}

#elif defined(__clang__) || defined(__GNUC__)

constexpr bool available = true;
constexpr std::size_t width = 4;
constexpr const char* backend_name = "vec4";

using NativeVec4f = float __attribute__((vector_size(16)));

struct Vec4f {
    NativeVec4f value;
};

inline Vec4f splat(float value) {
    return {NativeVec4f{value, value, value, value}};
}

inline Vec4f load_u(const float* source) {
    NativeVec4f value;
    std::memcpy(&value, source, sizeof(value));
    return {value};
}

inline void store_u(float* destination, Vec4f value) {
    std::memcpy(destination, &value.value, sizeof(value.value));
}

inline Vec4f operator+(Vec4f lhs, Vec4f rhs) {
    return {lhs.value + rhs.value};
}

inline Vec4f operator-(Vec4f lhs, Vec4f rhs) {
    return {lhs.value - rhs.value};
}

inline Vec4f operator*(Vec4f lhs, Vec4f rhs) {
    return {lhs.value * rhs.value};
}

#else

constexpr bool available = false;
constexpr std::size_t width = 1;
constexpr const char* backend_name = "scalar";

#endif

}  // namespace physics::simd
