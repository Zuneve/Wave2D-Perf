#pragma once

#if defined(__clang__) || defined(__GNUC__)
#define PHYSICS_RESTRICT __restrict__
#else
#define PHYSICS_RESTRICT
#endif
