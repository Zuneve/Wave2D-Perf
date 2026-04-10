# physicsProject

Исследовательский стенд для сравнения нескольких реализаций эволюции двумерной волновой функции во времени.

Сейчас в проекте уже есть:

- `naive_aos`: наивная реализация на `std::complex<float>` с AoS-раскладкой.
- `soa_*`: CPU-версия с раздельными массивами `real[]` / `imag[]`, padding по строкам и ручной SIMD-векторизацией.
  На `arm64` используется NEON, на `x86_64` используется SSE, иначе включается fallback через compiler vector extensions.
- `soa_*_threads`: многопоточный backend на `std::thread` и `std::barrier`, работает даже там, где OpenMP не установлен.
- `soa_*_omp`: тот же backend, но с опциональным OpenMP-разбиением по блокам строк.
- `cuda`: опциональный CUDA backend, который подключается только если CMake находит CUDA toolkit.

Модель выбрана простой и удобной для бенчмарка: явная схема для 2D time-dependent Schrodinger equation на пятиузловом stencil'е. Это хороший старт для исследования памяти, векторизации и GPU, даже если позже мы захотим перейти на более физически аккуратную схему.

## Сборка

```bash
cmake -S . -B build
cmake --build build -j
```

## Быстрый запуск

```bash
./build/physicsProject
./build/physicsProject --solver naive --nx 512 --ny 512 --steps 300
./build/physicsProject --solver soa --nx 1024 --ny 1024 --steps 500 --tile-x 256 --tile-y 32
./build/physicsProject --solver threads --threads 8 --nx 2048 --ny 2048 --steps 300
./build/physicsProject --solver omp --threads 8 --tile-y 16
./build/physicsProject --solver cuda
```

## Полезные аргументы

- `--solver all|naive|soa|threads|omp|cuda`
- `--nx`, `--ny`
- `--steps`, `--warmup`
- `--dt`, `--dx`, `--dy`
- `--mass`
- `--tile-x`, `--tile-y`
- `--threads`
- `--potential-strength`, `--packet-sigma`, `--packet-kx`, `--packet-ky`

## Дальше

Замечание по именам в выводе:

- `--solver soa` выбирает оптимизированный CPU backend, а в таблице он будет показан как конкретная реализация, например `soa_neon` на Apple Silicon или `soa_sse` на x86_64.
- `--solver threads` выбирает тот же SIMD kernel, но с многопоточным разбиением по непрерывным диапазонам строк.

Естественные следующие шаги:

1. Добавить более устойчивую схему времени, например split-step Fourier или Crank-Nicolson / ADI.
2. Вынести CPU SIMD на ISA-специфичные intrinsics и сравнить против compiler vector extensions.
3. Исследовать влияние разных стратегий разбиения на блоки и ложного sharing при OpenMP.
4. Доработать CUDA путь: shared memory tiling, pinned host memory, overlap transfer/compute, сравнение с unified memory.
