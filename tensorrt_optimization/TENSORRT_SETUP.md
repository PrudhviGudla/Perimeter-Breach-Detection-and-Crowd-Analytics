# TensorRT Installation Guide for Windows

Installing TensorRT on Windows can be tricky due to strict version matching and missing system DLLs. Follow this guide to avoid common "DLL not found" errors.

## Prerequisites
1.  **NVIDIA GPU** (RTX 30xx/40xx recommended).
2.  **CUDA Toolkit 12.x** installed.
    * Check version: `nvcc --version`

---

## Step 1: The "Hidden" Dependency (zlibwapi.dll)
TensorRT on Windows requires a specific compression library that is NOT included by default.

1.  Download **zlib123dllx64.zip** from [WinImage](http://www.winimage.com/zLibDll/zlib123dllx64.zip).
2.  Extract the zip.
3.  Copy `zlibwapi.dll` (from the `dll_x64` folder).
4.  Paste it into your CUDA bin directory:
    * `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin`

> **Note:** Without this file, TensorRT import will crash with `[WinError 126] The specified module could not be found`.

---

## Step 2: Download TensorRT
1.  Go to the [NVIDIA TensorRT Download Archive](https://developer.nvidia.com/tensorrt).
2.  Select **TensorRT 10.x.x GA**.
3.  **Critical:** Download the **ZIP Package** for Windows (e.g., `TensorRT-10.x.x.x...CUDA-12.x.zip`).
    * *Do NOT use the exe installer.*

---

## Step 3: Install & Path Configuration
1.  Extract the zip to a short path to avoid Windows 260-char limit errors.
    * **Recommended:** `C:\TensorRT`
2.  **Add to System PATH:**
    * Open "Edit the system environment variables".
    * Edit `Path` -> New -> Add `C:\TensorRT\lib`.
    * Edit `Path` -> New -> Add `C:\TensorRT\bin`.
    * *Restart your terminal/VS Code for this to apply.*

---

## Step 4: Install Python Bindings
1.  Navigate to the python folder:
    ```bash
    cd C:\TensorRT\python
    ```
2.  Install the wheel matching your Python version (e.g., `cp310` for Python 3.10):
    ```bash
    pip install tensorrt-*-cp310-none-win_amd64.whl
    ```
3.  Verify installation:
    ```python
    import tensorrt as trt
    print(trt.__version__)
    ```