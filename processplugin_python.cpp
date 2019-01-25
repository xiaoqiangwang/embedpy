#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "processplugin_python.h"

/* module/function object pointes */
static PyObject *pModule = NULL;
static PyObject *pInitFunc = NULL;
static PyObject *pCalcFunc = NULL;
static PyObject *pFreeFunc = NULL;

/* process plugin pythob module name, hence the the python filename */
const char *module_name = "processplugin";

/* wrap C array to numpy array */
PyObject *arrayToNumpy(void *data, size_t size_x, size_t size_y, int dtype)
{
    int ndims = 0;
    npy_intp dims[2] = {0, 0};

    if (size_x > 0) {
        ndims += 1;
        dims[0] = size_x;
        if (size_y > 0) {
            ndims += 1;
            dims[0] = size_y;
            dims[1] = size_x;
        }
    }
    return PyArray_SimpleNewFromData(ndims, dims, dtype, data);
}

void init_adapter(size_t size_x, size_t size_y, size_t size_z, int device) {
    /* initialize the Python interpreter and numpy if not yet */
    if (!Py_IsInitialized()) {
        Py_Initialize();
        if (_import_array() < 0)
            PyErr_Print();
    }

    /* import the process plugin python module if not yet */
    if (pModule == NULL)
        pModule = PyImport_ImportModule(module_name);
    if (pModule == NULL) {
        PyErr_Print();
        return;
    }

    /* get the plugin's entry functions */
    if (pInitFunc == NULL)
        pInitFunc = PyObject_GetAttrString(pModule, "init");
    if (pInitFunc == NULL) {
        PyErr_Print();
        return;
    }
    if (pCalcFunc == NULL)
        pCalcFunc = PyObject_GetAttrString(pModule, "calc");
    if (pCalcFunc == NULL) {
        PyErr_Print();
        return;
    }
    if (pFreeFunc == NULL)
        pFreeFunc = PyObject_GetAttrString(pModule, "free");
    if (pFreeFunc == NULL) {
        PyErr_Print();
        return;
    }

    /* call init entry */
    PyObject *result = PyObject_CallFunction(pInitFunc, "(nnni)", size_x, size_y, size_z, device);
    if (result == NULL) {
        PyErr_Print();
        return;
    }
}

void calc_adapter(int sizedata, struct feedback_data* data) {
    if (data == NULL || pCalcFunc == NULL)
        return;

    /* wrap input images as a list of numpy arrays */
    PyObject *waveinput = PyList_New(FEED_MAX_INPUT_WAVEFORMPARAMETER);
    if (waveinput == NULL) {
        PyErr_Print();
        return;
    }

    for (int i = 0; i < FEED_MAX_INPUT_WAVEFORMPARAMETER; i++) {
        if (data->waveinput[i] == NULL) {
            /* NULL pointer converts to None */
            Py_INCREF(Py_None);
            PyList_SetItem(waveinput, i, Py_None);
        } else {
            PyObject *waveform = arrayToNumpy(data->waveinput[i], data->waveform_x_input, data->waveform_y_input, NPY_USHORT);
            PyList_SetItem(waveinput, i, waveform);
        }
    }

    /* convert feedback_data structure as a Python dict */
    PyObject *pData = Py_BuildValue("{s:i,s:N,s:N,s:N,s:N,s:N,s:N,s:N,s:N,s:N,s:N}", 
        "device", data->device,

        "ai", arrayToNumpy(data->ai, FEED_MAX_INPUT_ANALOGPARAMETER, 0, NPY_DOUBLE),
        "bi", arrayToNumpy(data->bi, FEED_MAX_INPUT_BINARYPARAMETER, 0, NPY_UINT),
        "waveinput", waveinput,

        "ao", arrayToNumpy(data->ao, FEED_MAX_OUTPUT_ANALOGPARAMETER, 0, NPY_DOUBLE),
        "ao_valid", arrayToNumpy(data->ao_valid, FEED_MAX_OUTPUT_ANALOGPARAMETER, 0, NPY_SHORT),

        "bo", arrayToNumpy(data->bo, FEED_MAX_OUTPUT_BINARYPARAMETER, 0, NPY_UINT),
        "bo_valid", arrayToNumpy(data->bo_valid, FEED_MAX_OUTPUT_BINARYPARAMETER, 0, NPY_SHORT),

        "waveoutput", waveinput,

        "shadow_ao", arrayToNumpy(data->shadow_ao, FEED_MAX_INPUT_ANALOGPARAMETER, 0, NPY_DOUBLE),
        "shadow_bo", arrayToNumpy(data->shadow_bo, FEED_MAX_INPUT_BINARYPARAMETER, 0, NPY_UINT)
    );

    if (pData == NULL) {
        PyErr_Print();
        return;
    }

    PyObject *result = PyObject_CallFunction(pCalcFunc, "(iO)", sizedata, pData);
    if (result == NULL) {
        PyErr_Print();
    }

    /* check the processing results, for now print them out */
    PyObject_Print(result, stdout, 0);
    fprintf(stdout, "\n");

    Py_XDECREF(pData);
}

void free_adapter(int device) {
    if (pFreeFunc == NULL)
        return;

    PyObject *result = PyObject_CallFunction(pFreeFunc, "(i)", device);
    if (result == NULL) {
        PyErr_Print();
        return;
    }
}

