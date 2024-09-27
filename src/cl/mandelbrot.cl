#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void mandelbrot(__global float *results,
                         unsigned int width, unsigned int height,
                         float fromX, float fromY,
                         float sizeX, float sizeY,
                         unsigned int iters, int smoothing)
{
    {
        // TODO если хочется избавиться от зернистости и дрожания при интерактивном погружении, добавьте anti-aliasing:
        // грубо говоря, при anti-aliasing уровня N вам нужно рассчитать не одно значение в центре пикселя, а N*N значений
        // в узлах регулярной решетки внутри пикселя, а затем посчитав среднее значение результатов - взять его за результат для всего пикселя
        // это увеличит число операций в N*N раз, поэтому при рассчетах гигаплопс антиальясинг должен быть выключен
    }

    const unsigned int i = get_global_id(0);
    const unsigned int j = get_global_id(1);

    const float threshold = 256.0f;
    const float threshold2 = threshold * threshold;

    if (i >= width || j >= height)
        return;

    float x0 = fromX + (i + 0.5f) * sizeX / width;
    float y0 = fromY + (j + 0.5f) * sizeY / height;

    float x = x0;
    float y = y0;

    unsigned int iter = 0;
    for (; iter < iters; ++iter) {
        float xPrev = x;
        x = x * x - y * y + x0;
        y = 2.0f * xPrev * y + y0;
        if ((x * x + y * y) > threshold2) {
            break;
        }
    }
    float result = iter;
    if (smoothing && iter != iters) {
        result = result - log(log(sqrt(x * x + y * y)) / log(threshold)) / log(2.0f);
    }
// // anti-aliasing version
//    float result = 0.f;
//    const unsigned int aaLevel = 5;
//    unsigned int samples = aaLevel * aaLevel;
//
//    for (int m = 0; m < aaLevel; ++m) {
//        for (int n = 0; n < aaLevel; ++n) {
//            float x0 = fromX + (i + (m + 0.5f) / aaLevel) * sizeX / width;
//            float y0 = fromY + (j + (n + 0.5f) / aaLevel) * sizeY / height;
//
//            float x = x0;
//            float y = y0;
//
//            unsigned int iter = 0;
//            for (; iter < iters; ++iter) {
//                float xPrev = x;
//                x = x * x - y * y + x0;
//                y = 2.0f * xPrev * y + y0;
//                if ((x * x + y * y) > threshold2) {
//                    break;
//                }
//            }
//            float sampleResult = iter;
//            if (smoothing && iter != iters) {
//                sampleResult = sampleResult - log(log(sqrt(x * x + y * y)) / log(threshold)) / log(2.0f);
//            }
//            result += sampleResult / iters;
//        }
//    }
//    result /= samples;

    result = 1.0f * result / iters;
    results[j * width + i] = result;
}
