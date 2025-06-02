PYTHON3 := $(shell which python3 2>/dev/null)

PYTHON := python3

UNAME_S := $(shell uname -s)
UNAME_P := $(shell uname -p)
QRACK_PRESENT := $(wildcard qrack/.)

.PHONY: help
help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  build-deps         to build PennyLane-Qrack C++ dependencies"
	@echo "  install            to install PennyLane-Qrack"
	@echo "  wheel              to build the PennyLane-Qrack wheel"
	@echo "  dist               to package the source distribution"

.PHONY: build-deps
build-deps:
ifneq ($(OS),Windows_NT)
ifeq ($(QRACK_PRESENT),)
	git clone https://github.com/unitaryfund/qrack.git; cd qrack; git checkout f0738286f49fb0ffaeb46d6e5d2294dbfc7f6138; cd ..
endif
	mkdir -p qrack/build
ifeq ($(UNAME_S),Linux)
ifeq ($(UNAME_P),x86_64)
	cd qrack/build; cmake -DENABLE_RDRAND=OFF -DENABLE_DEVRAND=ON -DENABLE_OPENCL=OFF -DENABLE_CUDA=ON -DQBCAPPOW=8 ..; make qrack_pinvoke
else
	cd qrack/build; cmake -DENABLE_RDRAND=OFF -DENABLE_DEVRAND=ON -DENABLE_COMPLEX_X2=OFF -DENABLE_SSE3=OFF -DENABLE_OPENCL=OFF -DENABLE_CUDA=ON -DQBCAPPOW=8 ..; make qrack_pinvoke
endif
endif
ifeq ($(UNAME_S),Darwin)
ifeq ($(UNAME_P),x86_64)
	cd qrack/build; cmake -DENABLE_OPENCL=OFF -DENABLE_CUDA=ON -DQBCAPPOW=8 ..; make qrack_pinvoke
else
	cd qrack/build; cmake -DENABLE_RDRAND=OFF -DENABLE_COMPLEX_X2=OFF -DENABLE_SSE3=OFF -DENABLE_OPENCL=OFF -DENABLE_CUDA=ON -DQBCAPPOW=8 ..; make qrack_pinvoke
endif
endif
endif
	mkdir pyqrack/qrack_system/qrack_lib; cp qrack/build/libqrack_pinvoke.* pyqrack/qrack_system/qrack_lib/; cd ../../..

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
