![Unit Test Status](https://github.com/ThomasHelfer/HigherOrderInterpolation3DTorch/actions/workflows/actions.yml/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)

# PyInterpX - Higher Order Interpolation in 3D for Torch
<p align="center">
    <img src="https://github.com/ThomasHelfer/PyInterpX/blob/main/img/logo_cropped.png" alt="no alignment" width="25%" height="auto"/>
</p>

## Overview

PyInterpX is a compact library designed for advanced 3D interpolation using higher order polynomial bases, which is not currently supported by PyTorch's `torch.nn.functional.interpolate()` method. This enhancement allows for more precise and customized interpolation processes in 3D spaces, catering to specialized applications requiring beyond-linear data manipulation.

## Quick Start

To get started with PyInterpX:

1. Install the library using pip:

    ```bash
    pip install pyinterpx
    ```

2. Import `interp` from PyInterpX and PyTorch in your script or notebook:

    ```python
    from pyinterpx.Interpolation import interp
    import torch
    ```

3. Utilize the interpolation function with a 6x6x6 kernel, polynomials up to the third power, and 25 channels:

    ```python
    points, power, channels = 6, 3, 25
    Interp = interp(points, power, channels)
    x = torch.rand(2, 25, 10, 10, 10)
    Interp(x)
    ```

## Key Features

- **Fast**: Optimized for high performance across any device.

![Performance Comparison](https://github.com/ThomasHelfer/HigherOrderInterpolation3DTorch/blob/main/img/Comparison.png "Performance Comparison")

- **CPU and GPU Compatible**: Functions seamlessly on both CPU and GPU environments.

    ```python
    points, power, channels = 6, 3, 25
    # Running on GPU for even faster computations 
    interp = interp(points, power, channels, device="cuda:0")
    ```

- **Precise**: Supports various data types for precise computation.

    ```python
    points, power, channels = 6, 3, 25
    # Using double for more precision 
    interp = interp(points, power, channels, dtype=torch.double)
    ```

- **Integrated with PyTorch**: Easily integrates within the PyTorch ecosystem.

    ```python
    # A simple model which uses interpolation at some layer
    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            points, power, channels = 6, 3, 25
            # Setting up interpolation 
            self.interpolation = interp(points, power, channels)

            self.convs = torch.nn.Sequential(
                torch.nn.Conv3d(25, 64, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),
            )

        def forward(self, x):
            x = self.convs(x)
            x = self.interpolation(x)
            return x
    ```
- Choose simply the **grid alignment** you like.
  
    ```python
    points, power, channels = 6, 3, 25
    interp = interp(points, power, channels, dtype=torch.double,align_corners = False)
    ```
    <p align="center">
        <img src="https://github.com/ThomasHelfer/HigherOrderInterpolation3DTorch/blob/main/img/no_align.png" alt="no alignment" width="50%" height="auto"/>
    </p>
    or if you do not want to have any aligment with the input grid
    
    ```python
    points, power, channels = 6, 3, 25
    interp = interp(points, power, channels, dtype=torch.double,align_corners = True)
    ```
    
    <p align="center">
        <img src="https://github.com/ThomasHelfer/HigherOrderInterpolation3DTorch/blob/main/img/align.png" alt="aligned" width="50%" height="auto"/>
    </p>

- Choose the enhacement **factor** you like
    ```python
    factor = 4
    points, power, channels = 6, 3, 25
    interp = interp(points, power, channels, dtype=torch.double,align_corners = False,factor = factor)
    ```
    <p align="center">
        <img src="https://github.com/ThomasHelfer/HigherOrderInterpolation3DTorch/blob/main/img/interpolation_grid_zoomed_factor_4.png" alt="no alignment" width="50%" height="auto"/>
    </p>
    
     ```python
    factor = 16
    points, power, channels = 6, 3, 25
    interp = interp(points, power, channels, dtype=torch.double,align_corners = False,factor = factor)
    ```
    <p align="center">
        <img src="https://github.com/ThomasHelfer/HigherOrderInterpolation3DTorch/blob/main/img/interpolation_grid_zoomed_factor_16.png" alt="no alignment" width="50%" height="auto"/>
    </p>    
    
### Prerequisites

Before installing PyInterpX, ensure you meet the following prerequisites:
- Python 3.8 or higher
- pip package manager

## License

PyInterpX is open-sourced under the MIT License. For more details, see the LICENSE file.

## Contact

For inquiries or support, reach out to Thomas Helfer at thomashelfer@live.de.
