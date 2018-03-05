$ python3 setup.py build
* Create .so file from Id.cpp (C++ source code)

$ cp build/lib.{arch}/Id.{arch}.so .
* Fill in your existed {arch} depending on your architecture and
* version of python

$ mv Id.{arch}.so Id.so
* Just rename to Id.so

$ python3 main.py test.png lenet_deploy.prototxt lenet_iter_100000.caffemodel
* Enjoy!

* Browse main.py, see that I caught and stored content into 3 vars:
* dob, name, addr
* and not just printing to stdout
