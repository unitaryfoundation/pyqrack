#!/bin/bash

PLATFORMS=("manylinux_2_35_x86_64" "linux_aarch64" "linux_armv7l" "manylinux2014_x86_64" "macosx_10_4_universal2" "win_amd64")
LIB_FOLDERS=("pyqrack/qrack_system/qrack_lib/Linux/2_35" "pyqrack/qrack_system/qrack_lib/Linux/ARM64" "pyqrack/qrack_system/qrack_lib/Linux/ARMv7" "pyqrack/qrack_system/qrack_lib/Linux/x86_64" "pyqrack/qrack_system/qrack_lib/Mac/M2" "pyqrack/qrack_system/qrack_lib/Windows/x86_64")
CL_FOLDERS=("pyqrack/qrack_cl_precompile/Linux/2_35" "pyqrack/qrack_cl_precompile/Linux/ARM64" "pyqrack/qrack_cl_precompile/Linux/ARMv7" "pyqrack/qrack_cl_precompile/Linux/x86_64" "pyqrack/qrack_cl_precompile/Mac/M2" "pyqrack/qrack_cl_precompile/Windows/x86_64")

TEMP_DIR="temp_pyqrack"

for i in "${!PLATFORMS[@]}"; do
    platform="${PLATFORMS[$i]}"
    lib_folder="${LIB_FOLDERS[$i]}"
    cl_folder="${CL_FOLDERS[$i]}"

    # Make a temporary copy of the PyQrack project directory
    cp -r ../pyqrack "$TEMP_DIR"

    # Remove unwanted folders for the specific platform
    for folder in "${LIB_FOLDERS[@]}"; do
        if [[ "$folder" != "$lib_folder" ]]; then
            rm -rf "$TEMP_DIR/$folder"
        fi
    done

    for folder in "${CL_FOLDERS[@]}"; do
        if [[ "$folder" != "$cl_folder" ]]; then
            rm -rf "$TEMP_DIR/$folder"
        fi
    done

    # Build the wheel for the specific platform
    cd "$TEMP_DIR"
    python3 setup.py bdist_wheel --plat-name="$platform"

    # Move the wheel to the dist directory
    mkdir -p ../dist
    mv dist/* ../dist/

    # Clean up the temporary directory
    cd ..
    rm -rf "$TEMP_DIR"
done

