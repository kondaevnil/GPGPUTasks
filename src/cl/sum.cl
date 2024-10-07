#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define VPW 32
#define WGS 64

__kernel void sum1(__global const unsigned int *a,
                   __global unsigned int *sum,
                   unsigned int n)
{
    const unsigned int ind = get_global_id(0);
    if (ind < n) {
        atomic_add(sum, a[ind]);
    }
}

__kernel void sum2(__global const unsigned int *a,
                   __global unsigned int *sum,
                   unsigned int n)
{
    const unsigned int gid = get_global_id(0);

    unsigned int res = 0;
    for (int i = 0; i < VPW; i++) {
        unsigned int ind = gid * VPW + i;
        if (ind < n) {
            res += a[ind];
        }
    }

    atomic_add(sum, res);
}

__kernel void sum3(__global const unsigned int *a,
                   __global unsigned int *sum,
                   unsigned int n)
{
    const unsigned int lid = get_local_id(0);
    const unsigned int wid = get_group_id(0);
    const unsigned int grs = get_local_size(0);

    unsigned int res = 0;
    for (unsigned int i = VPW * wid * grs + lid; i < VPW * (wid + 1) * grs; i += grs) {
        if (i < n) res += a[i];
    }

    atomic_add(sum, res);
}

__kernel void sum4(__global const unsigned int *a,
                   __global unsigned int *sum,
                   unsigned int n)
{
    const unsigned int gid = get_global_id(0);
    const unsigned int lid = get_local_id(0);

    __local unsigned int buf[WGS];
    buf[lid] = a[gid];

    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        unsigned int res = 0;
        for (int i = 0; i < WGS; i++) {
            res += buf[i];
        }
        atomic_add(sum, res);
    }
}

__kernel void sum5(__global const unsigned int *a,
                   __global unsigned int *sum,
                   unsigned int n)
{
    const unsigned int gid = get_global_id(0);
    const unsigned int lid = get_local_id(0);

    __local unsigned int buf[WGS];
    buf[lid] = a[gid];

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = WGS; i > 1; i /= 2) {
        if (2 * lid < i) {
            unsigned int b = buf[lid];
            unsigned int c = buf[lid + i / 2];
            buf[lid] = b + c;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) atomic_add(sum, buf[0]);
}