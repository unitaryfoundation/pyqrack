PYTHON3 := $(shell which python3 2>/dev/null)

PYTHON := python3

UNAME_S := $(shell uname -s)
UNAME_P := $(shell uname -p)
QRACK_PRESENT := $(wildcard qrack/.)

ifeq ("$(wildcard /usr/local/bin/cmake)", "/usr/local/bin/cmake")
CMAKE_L := /usr/local/bin/cmake
else
ifeq ("$(wildcard /usr/bin/cmake)", "/usr/bin/cmake")
CMAKE_L := /usr/bin/cmake
else
CMAKE_L := cmake
endif
endif

.PHONY: help
help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  build-deps         to build PennyLane-Qrack C++ dependencies"
	@echo "  install            to install PennyLane-Qrack"
	@echo "  wheel              to build the PennyLane-Qrack wheel"
	@echo "  dist               to package the source distribution"

.PHONY: build-deps
build-deps:
	rm -rf pyqrack/qrack_system/qrack_lib
	rm -rf pyqrack/qrack_system/qrack_cl_precompile
ifneq ($(OS),Windows_NT)
ifeq ($(QRACK_PRESENT),)
	git clone https://github.com/unitaryfund/qrack.git; cd qrack; git checkout 4154230f4e6ccbaf44d1708a5aff59a578ab7119; cd ..
endif
	mkdir -p qrack/build
ifeq ($(UNAME_S),Linux)
ifneq ($(filter $(UNAME_P),x86_64 i386),)
	cd qrack/build; $(CMAKE_L) -DENABLE_RDRAND=OFF -DENABLE_DEVRAND=ON -DQBCAPPOW=8 ..; make qrack_pinvoke qrack_cl_precompile
else
	cd qrack/build; $(CMAKE_L) -DENABLE_RDRAND=OFF -DENABLE_DEVRAND=ON -DENABLE_COMPLEX_X2=OFF -DENABLE_SSE3=OFF -DQBCAPPOW=8 ..; make qrack_pinvoke qrack_cl_precompile
endif
endif
ifeq ($(UNAME_S),Darwin)
ifneq ($(filter $(UNAME_P),x86_64 i386),)
	cd qrack/build; cmake -DENABLE_OPENCL=OFF -DQBCAPPOW=8 -DBoost_INCLUDE_DIR=/opt/homebrew/include -DBoost_LIBRARY_DIRS=/opt/homebrew/lib ..; make qrack_pinvoke qrack_cl_precompile
else
	cd qrack/build; cmake -DENABLE_OPENCL=OFF -DENABLE_RDRAND=OFF -DENABLE_COMPLEX_X2=OFF -DENABLE_SSE3=OFF -DQBCAPPOW=8 -DBoost_INCLUDE_DIR=/opt/homebrew/include -DBoost_LIBRARY_DIRS=/opt/homebrew/lib ..; make qrack_pinvoke qrack_cl_precompile
endif
endif
endif
	mkdir pyqrack/qrack_system/qrack_lib; cp qrack/build/libqrack_pinvoke.* pyqrack/qrack_system/qrack_lib/; cd ../../..
	mkdir pyqrack/qrack_system/qrack_cl_precompile; cp qrack/build/qrack_cl_precompile pyqrack/qrack_system/qrack_cl_precompile/; cd ../../..

.PHONY: install
install:
ifndef PYTHON3
	@echo "To install PyQrack you need to have Python 3 installed"
endif
	$(PYTHON) setup.py install

.PHONY: wheel
wheel:
	$(PYTHON) setup.py bdist_wheel

.PHONY: dist
dist:
	$(PYTHON) setup.py sdist
