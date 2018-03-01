from distutils.core import setup, Extension

setup(name='Id', version='0.01', ext_modules=[Extension('Id', ['Id.cpp', 'IdCard.cpp'])])