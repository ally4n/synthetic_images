
import numpy as np
from math import pi

def rotated_blob(x, y, z, original_blob, angle):
    # Rotate points around z-axis
    x_rot = x * np.cos(angle) - y * np.sin(angle)
    y_rot = x * np.sin(angle) + y * np.cos(angle)
    return original_blob(x_rot, y_rot, z)

def scaled_blob(x, y, z, original_blob, scale):
    return original_blob(x/scale, y/scale, z/scale)

def create_random_blob(center, max_boundaries, volume_shape, pixel_spacing, slice_thickness):
    
    def blob_function(x, y, z):
        num_gaussians = 5
        amplitudes = np.random.rand(num_gaussians) * 0.5 + 0.5
        sigmas = np.random.rand(num_gaussians, 3) * np.array(max_boundaries) * 0.3  # Increased from 0.2 to 0.3
        centers = np.random.rand(num_gaussians, 3) * np.array(max_boundaries) * 2 - np.array(max_boundaries)
        
        result = 0
        for i in range(num_gaussians):
            dx = x - (center[0] + centers[i, 0])
            dy = y - (center[1] + centers[i, 1])
            dz = z - (center[2] + centers[i, 2])
            result += amplitudes[i] * np.exp(-(dx**2 / (2 * sigmas[i, 0]**2) + 
                                               dy**2 / (2 * sigmas[i, 1]**2) + 
                                               dz**2 / (2 * sigmas[i, 2]**2)))
        return result > 0.2  # Changed from 0.3 to 0.2 to make the blob even larger

    x, y, z = np.ogrid[:volume_shape[0], :volume_shape[1], :volume_shape[2]]
    x = (x - center[0]) * slice_thickness
    y = (y - center[1]) * pixel_spacing[1]
    z = (z - center[2]) * pixel_spacing[0]
    
    blob = blob_function(x, y, z).astype(np.int16) * 2000
    return blob, blob_function

def create_sphere(radius, center, volume_shape, pixel_spacing, slice_thickness):
    x, y, z = np.ogrid[:volume_shape[0], :volume_shape[1], :volume_shape[2]]
    x = (x - center[0]) * slice_thickness
    y = (y - center[1]) * pixel_spacing[1]
    z = (z - center[2]) * pixel_spacing[0]
    dist_from_center = np.sqrt(x*x + y*y + z*z)
    sphere = (dist_from_center <= radius).astype(np.int16) * 1000
    return sphere

def cube_blob(x, y, z, side_length=2):
    return (abs(x) <= side_length/2) & (abs(y) <= side_length/2) & (abs(z) <= side_length/2)

def cylinder_blob(x, y, z, radius=1, height=2):
    return (x**2 + y**2 <= radius**2) & (abs(z) <= height/2)

def ellipsoid_blob(x, y, z, a=1, b=2, c=3):
    return (x**2/a**2 + y**2/b**2 + z**2/c**2 <= 1)

def monte_carlo_volume(blob_function, center, max_boundaries, pixel_spacing, slice_thickness, num_samples=1000000):
    center = np.array(center)
    if center.shape != (3,):
        raise ValueError("Center must be a 3D point (x, y, z)")

    # Generate random points within the bounding box
    points = np.random.rand(num_samples, 3) * np.array(max_boundaries) * 2 - np.array(max_boundaries)
    
    # Apply blob function
    inside_count = np.sum(blob_function(points[:, 0], points[:, 1], points[:, 2]))
    
    # Calculate the volume, considering pixel spacing and slice thickness
    voxel_volume = pixel_spacing[0] * pixel_spacing[1] * slice_thickness
    bounding_box_volume = np.prod(np.array(max_boundaries) * 2) * voxel_volume
    object_volume = (inside_count / num_samples) * bounding_box_volume
    
    return object_volume