import numpy as np
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_and_preprocess_data(file_path):
    try:
        data = np.load(file_path)
        points = data['points']
        print(f"Loaded {len(points)} points from the data file")
        print(f"Point cloud shape: {points.shape}")
        return points
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def analyze_shape_characteristics(points):
    # Calculate bounding box
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    dimensions = max_coords - min_coords
    
    # Calculate center and average distance from center
    center = np.mean(points, axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    avg_distance = np.mean(distances)
    std_distance = np.std(distances)
    
    print("\nShape Characteristics:")
    print(f"Dimensions (x,y,z): {dimensions}")
    print(f"Center point: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
    print(f"Average distance from center: {avg_distance:.3f}")
    print(f"Standard deviation of distances: {std_distance:.3f}")
    
    # Calculate aspect ratios
    ranges = np.max(points, axis=0) - np.min(points, axis=0)
    min_range = np.min(ranges)
    aspect_ratios = ranges / min_range
    print(f"Aspect ratios (x:y:z): {aspect_ratios[0]:.3f}:{aspect_ratios[1]:.3f}:{aspect_ratios[2]:.3f}")
    
    return center, distances

def identify_deformations(points, distances):
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    
    # Define thresholds for potential deformations
    outer_threshold = mean_dist + 2 * std_dist
    inner_threshold = mean_dist - 2 * std_dist
    
    # Identify points that deviate significantly from the mean
    deformed_points = np.logical_or(
        distances > outer_threshold,
        distances < inner_threshold
    )
    
    deformation_count = np.sum(deformed_points)
    print(f"\nDeformation Analysis:")
    print(f"Mean distance from center: {mean_dist:.3f}")
    print(f"Distance standard deviation: {std_dist:.3f}")
    print(f"Outer threshold: {outer_threshold:.3f}")
    print(f"Inner threshold: {inner_threshold:.3f}")
    print(f"Potential deformation points: {deformation_count} ({deformation_count/len(points)*100:.2f}%)")
    
    return deformed_points

def save_visualization(points, deformed_points, filename='shape_analysis.png'):
    fig = plt.figure(figsize=(20, 5))
    
    # Create views from different angles
    views = [(30, 45), (90, 0), (0, 90), (0, 0)]
    titles = ['Isometric View', 'Side View', 'Top View', 'Front View']
    
    for i, (elev, azim) in enumerate(views):
        ax = fig.add_subplot(1, 4, i+1, projection='3d')
        
        # Plot normal points in blue
        ax.scatter(points[~deformed_points, 0], 
                  points[~deformed_points, 1], 
                  points[~deformed_points, 2],
                  c='blue', alpha=0.6, s=1, label='Normal')
        
        # Plot potentially deformed points in red
        ax.scatter(points[deformed_points, 0], 
                  points[deformed_points, 1], 
                  points[deformed_points, 2],
                  c='red', alpha=0.8, s=1, label='Potential Deformation')
        
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(titles[i])
        
        if i == 0:  # Only add legend to first plot
            ax.legend()
        
        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nVisualization saved as {filename}")

def analyze_pca(points):
    pca = PCA(n_components=3)
    pca.fit(points)
    
    print("\nPrincipal Component Analysis:")
    for i, ratio in enumerate(pca.explained_variance_ratio_):
        print(f"PC{i+1} explains {ratio*100:.2f}% of variance")
    
    # Check for symmetry/asymmetry
    variance_ratios = pca.explained_variance_ratio_
    if np.max(variance_ratios) - np.min(variance_ratios) < 0.2:
        print("Shape appears to be relatively symmetric")
    else:
        print("Shape shows significant asymmetry along principal components")
    
    return pca

def main():
    # Load data
    points = load_and_preprocess_data('3d_shape_points_data.npz')
    if points is None:
        return None
    
    # Analyze shape characteristics
    center, distances = analyze_shape_characteristics(points)
    
    # Perform PCA analysis
    pca = analyze_pca(points)
    
    # Identify deformations
    deformed_points = identify_deformations(points, distances)
    
    # Save visualization
    save_visualization(points, deformed_points)

if __name__ == "__main__":
    main()