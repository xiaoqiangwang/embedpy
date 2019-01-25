#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "processplugin_python.h"

/* test program to drive the entry functions */
int main(int argc, char **argv)
{
    const size_t size_x=100, size_y=100, size_z=1;
    int device = 2;
    struct feedback_data demo_data = {0};

    /* demo image showing a centered gaussian spot */
    double offset = 10, amplitude = 100, x0 = size_x / 2, y0 = size_y / 2, sigmax = 30, sigmay=50;
    unsigned short image[size_x * size_y];
    for (size_t j = 0; j < size_y; j++)
        for (size_t i = 0; i < size_x; i++) {
            image[j * size_x + i] = (unsigned short) (offset + rand() % 10 + amplitude * exp(-0.5 * pow(i-x0, 2) / pow(sigmax, 2) - 0.5 * pow(j-y0, 2) / pow(sigmay, 2)));
        }
 
    /* fill in demo_data */
    demo_data.device = device;
    demo_data.ai[1] = 1;
    demo_data.waveform_x_input = size_x;
    demo_data.waveform_y_input = size_y;
    demo_data.waveinput[0] = image;

    /* call the entry functions in turn */
    init_adapter(size_x, size_y, size_z, device);
    calc_adapter(device, &demo_data);
    free_adapter(device);

    return 0;
}
