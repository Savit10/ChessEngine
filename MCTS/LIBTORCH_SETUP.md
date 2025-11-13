# LibTorch Setup Guide

This guide explains how to set up LibTorch (PyTorch C++ API) for neural network inference in the MCTS chess engine.

## Step 1: Download LibTorch

1. Go to https://pytorch.org/get-started/locally/
2. Select:
   - **PyTorch Build**: Stable (recommended)
   - **Your OS**: macOS
   - **Package**: LibTorch
   - **Language**: C++
   - **Compute Platform**: CPU (or CUDA if you have GPU support)

3. Download the appropriate version for your system:
   - **For Apple Silicon (M1/M2/M3)**: Download the arm64 version
   - **For Intel Mac**: Download the x86_64 version

4. Extract the downloaded ZIP file:
   ```bash
   cd ~
   unzip libtorch-macos-*.zip
   ```

   This will create a `libtorch` directory in your home folder.

## Step 2: Verify Installation

Check that libtorch was extracted correctly:
```bash
ls ~/libtorch
```

You should see directories like:
- `include/` - Header files
- `lib/` - Library files
- `share/` - CMake files

## Step 3: Set Environment Variable (Optional)

You can set `LIBTORCH_PATH` in your shell profile:
```bash
# For zsh (default on macOS)
echo 'export LIBTORCH_PATH=$HOME/libtorch' >> ~/.zshrc
source ~/.zshrc

# For bash
echo 'export LIBTORCH_PATH=$HOME/libtorch' >> ~/.bash_profile
source ~/.bash_profile
```

Or set it when building:
```bash
LIBTORCH_PATH=~/libtorch make Chess
```

## Step 4: Build the Project

The Makefile will automatically detect libtorch if it's in `~/libtorch`:

```bash
cd MCTS
make Chess
```

If libtorch is in a different location, specify it:
```bash
LIBTORCH_PATH=/path/to/libtorch make Chess
```

## Step 5: Place Model File

Copy your `aznet_traced.pt` model file to the project root or specify its path when running:

```bash
./chess_mcts /path/to/aznet_traced.pt
```

## Troubleshooting

### Error: "libtorch not found"
- Make sure you extracted libtorch to `~/libtorch` or set `LIBTORCH_PATH`
- Check that the directory exists: `ls ~/libtorch`

### Error: "cannot find -ltorch"
- Verify libtorch libraries are in `$(LIBTORCH_PATH)/lib`
- Check that you downloaded the correct architecture (arm64 vs x86_64)

### Error: "dyld: Library not loaded"
- Set the library path: `export DYLD_LIBRARY_PATH=$HOME/libtorch/lib:$DYLD_LIBRARY_PATH`
- Or use rpath (already set in Makefile)

### Model loading errors
- Ensure `aznet_traced.pt` is a valid TorchScript model
- Check that the model was traced correctly in Python
- Verify the model path is correct

## Alternative: Use CMake

If you prefer CMake, create a `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(chess_mcts)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)

add_executable(chess_mcts
    Chess/main.cpp
    Chess/Chess.cpp
    src/mcts.cpp
    src/JobScheduler.cpp
    src/neural_network.cpp
)

target_link_libraries(chess_mcts "${TORCH_LIBRARIES}")
set_property(TARGET chess_mcts PROPERTY CXX_STANDARD 17)
```

Then build:
```bash
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=~/libtorch ..
make
```

