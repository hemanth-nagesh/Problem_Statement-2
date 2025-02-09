## Problem_Statement-2
# 3D Shape Analysis and Deformation Detection

## Description
This project is developed for the UST and provides tools for analyzing 3D point cloud data to detect and visualize deformations in machine components. It processes point cloud data stored in NPZ format, performs statistical analysis to identify potential deformations, and generates multi-view visualizations of the results.

## Prerequisites
- Python 3.x
- Required packages:
  ```
  numpy
  matplotlib
  scikit-learn
  ```

## Installation
1. Clone the repository:
   ```bash
   git clone [your-repository-url]
   ```

2. Install required packages:
   ```bash
   pip install numpy matplotlib scikit-learn
   ```

## Usage
1. Place your point cloud data file (3d_shape_points_data.npz) in the project directory

2. Run the analysis:
   ```bash
   python shape_analysis.py
   ```

3. The program will generate:
   - Detailed shape analysis in the console
   - Visualization saved as 'shape_analysis.png'

## Input Data Format
- The input file should be a .npz file containing 3D point cloud data
- Points should be stored under the key 'points'

## Output
### Console Output
- Shape characteristics (dimensions, center point)
- PCA analysis results
- Deformation analysis statistics

### Visual Output
- Multi-view visualization showing:
  - Normal points (blue)
  - Potential deformations (red)
  - Views: isometric, side, top, and front

## Contributing
Feel free to fork the repository and submit pull requests.

## License
personal use only

## Contact
here

## Acknowledgments
- Built using NumPy, Matplotlib, and Scikit-learn libraries
