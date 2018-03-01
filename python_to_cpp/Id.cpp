#include <python3.5/Python.h>
#include "IdCard.h"

// const char *g_name = "Nguyen Duc Minh";
// const char *g_addr = "So 151C Phao Dai Lang";
// const char *g_dob = "03-07-1996";

IdCard card1("Nguyễn Đức Minh", "Hà Nội", "03-07-1996");

const char *g_name;
const char *g_addr;
const char *g_dob;

static PyObject* get_name(PyObject* self)
{
    return Py_BuildValue("s", card1.get_name().c_str());
}

static PyObject* get_dob(PyObject* self)
{
    return Py_BuildValue("s", card1.get_dob().c_str());
}

static PyObject* get_addr(PyObject* self)
{
    return Py_BuildValue("s", card1.get_addr().c_str());
}

static PyMethodDef module_methods[] = {
    {"get_name", (PyCFunction) get_name, METH_NOARGS, NULL},
    {"get_addr", (PyCFunction) get_addr, METH_NOARGS, NULL},
    {"get_dob", (PyCFunction) get_dob, METH_NOARGS, NULL},
    {NULL}
};

static struct PyModuleDef Id =
{
    PyModuleDef_HEAD_INIT,
    "ID Card Recognition in C++", /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,   /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    module_methods
};

PyMODINIT_FUNC PyInit_Id(void)
{
    return PyModule_Create(&Id);
}

int main(int argc, char **argv)
{
    IdCard card1;
    g_name = card1.get_name().c_str();
    g_addr = card1.get_addr().c_str();
    g_dob = card1.get_dob().c_str();

    std::cout << "name: " << g_name << std::endl;
    std::cout << "addr: " << g_addr << std::endl;
    std::cout << "dob: " << g_dob << std::endl;
}