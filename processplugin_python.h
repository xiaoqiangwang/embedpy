#ifndef PROCESSPLUGIN_PYTHON_H
#define PROCESSPLUGIN_PYTHON_H

#ifdef _WIN32
    #ifdef PROCESSPLUGIN_PYTHON_EXPORT
        #define PROCESSPLUGIN_PYTHON_SHARE __declspec(dllexport)
    #else
        #define PROCESSPLUGIN_PYTHON_SHARE __declspec(dllimport)
    #endif
#else
    #define PROCESSPLUGIN_PYTHON_SHARE
#endif

#define FEED_MAX_INPUT_ANALOGPARAMETER 80
#define FEED_MAX_INPUT_BINARYPARAMETER 80
#define FEED_MAX_INPUT_WAVEFORMPARAMETER 4
#define FEED_MAX_OUTPUT_ANALOGPARAMETER 80
#define FEED_MAX_OUTPUT_BINARYPARAMETER 80
#define FEED_MAX_OUTPUT_WAVEFORMPARAMETER 4

#include <stddef.h>

struct feedback_data {
    int device;
    double ai[FEED_MAX_INPUT_ANALOGPARAMETER];
    unsigned int bi[FEED_MAX_INPUT_BINARYPARAMETER];
    unsigned short *waveinput[FEED_MAX_INPUT_WAVEFORMPARAMETER];
    size_t waveform_x_input;
    size_t waveform_y_input;
    size_t waveform_inputsize;
    double ao[FEED_MAX_OUTPUT_ANALOGPARAMETER];
    short ao_valid[FEED_MAX_OUTPUT_ANALOGPARAMETER];
    unsigned int bo[FEED_MAX_OUTPUT_BINARYPARAMETER];
    short bo_valid[FEED_MAX_OUTPUT_BINARYPARAMETER];
    double waveoutput[FEED_MAX_OUTPUT_WAVEFORMPARAMETER];
    void *algorithemdata;
    double shadow_ao[FEED_MAX_INPUT_ANALOGPARAMETER];
    unsigned int shadow_bo[FEED_MAX_INPUT_BINARYPARAMETER];
};

#ifdef __cplusplus
extern "C" {
#endif

PROCESSPLUGIN_PYTHON_SHARE void init_adapter(size_t size_x, size_t size_y, size_t size_z, int device);
PROCESSPLUGIN_PYTHON_SHARE void calc_adapter(int sizedata, feedback_data* data);
PROCESSPLUGIN_PYTHON_SHARE void free_adapter(int device);

#ifdef __cplusplus
}
#endif

#endif
