# Seam Carving Repository

This repository contains implementations of the Seam Carving algorithm for image resizing while preserving important content. Seam Carving intelligently removes or inserts pixels along the least significant seams in an image. This repository provides different implementations, including sequential, parallel (multi-threaded CPU), and GPU-based solutions.

## Files Overview

### 1. `seq_seamCarving.py`
**(Sequential Seam Carving Implementation)**
- Implements seam carving using a sequential approach.
- Uses Sobel filters to calculate the energy of an image.
- Finds the optimal seam to remove using dynamic programming.
- Removes seams iteratively to achieve the desired resizing.
- Highlights removed seams in red for visualization.
- Can be executed via command line:  
  ```sh
  python seq_seamCarving.py <image_path> <scale>
  ```
- **Runtime Performance**:  
  - **Low-dimension image (280 × 390 × 3)**: 33 min 20 sec  
  - **High-dimension image (349 × 1182 × 3)**: Over 30 min (did not finish)  
- Outputs:  
  - `output.jpg` (resized image)  
  - `seam_highlighted.jpg` (image with highlighted removed seams)  

### 2. `gpu_seamCarving.ipynb`
**(GPU-Accelerated Seam Carving using CUDA)**
- Uses GPU parallelization to speed up seam carving.
- Offloads energy computation and seam removal to the GPU.
- Designed to run on systems with CUDA-enabled GPUs.
- Provides a Jupyter Notebook interface for interactive testing.
- **Runtime Performance**:  
  - **Low-dimension image**: 20 sec  
  - **High-dimension image**: 1 min 30 sec  

### 3. `GPU_only_DP.ipynb`
**(GPU-Only Dynamic Programming for Seam Carving)**
- Focuses solely on using GPU for the dynamic programming part of seam carving.
- Compares performance gains of GPU-based DP over CPU-based DP.
- Suitable for benchmarking and performance analysis.
-  **Runtime Performance**:  
  - **Low-dimension image**: 8 min 2 sec (2-core parallel DP)  
  - **High-dimension image**: 12 min 3 sec (2-core parallel DP) 

### 4. `parallel using multi thread cpu.ipynb`
**(Parallel Seam Carving Using Multi-Threaded CPU Processing)**
- Implements parallel processing for seam carving using multi-threading.
- Optimizes energy calculation and seam removal using CPU threads.
- Provides performance analysis comparing single-threaded vs. multi-threaded execution.
- Useful for systems without a GPU but with multi-core processors.
- **Runtime Performance**:  
  - **Low-dimension image**: 12 min 12 sec (2-core parallel DP)  
  - **High-dimension image**: 20 min 3 sec (2-core parallel DP)  

## Usage Instructions
1. Run the `seq_seamCarving.py` script for basic sequential seam carving.
2. Use `gpu_seamCarving.ipynb` for GPU-accelerated seam carving.
3. Try `GPU_only_DP.ipynb` if you want to analyze GPU-optimized dynamic programming.
4. Run `parallel using multi thread cpu.ipynb` to test multi-threaded CPU-based seam carving.
5. Use `sheep-on-a-meadow.jpg` or any other image for testing.

## Future Improvements
- Implement hybrid CPU-GPU seam carving for optimal performance.
- Add support for inserting seams to enlarge images.
- Experiment with different energy functions for improved results.

---
Feel free to contribute or report any issues! 🚀
