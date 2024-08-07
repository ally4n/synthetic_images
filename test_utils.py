from object_utils import *

def convergence_test(blob_function, center, max_boundaries, pixel_spacing=(1, 1), slice_thickness=1):
    sample_sizes = [10000, 100000, 1000000, 10000000]
    for samples in sample_sizes:
        volume = monte_carlo_volume(blob_function, center, max_boundaries, pixel_spacing, slice_thickness, num_samples=samples)
        print(f"Samples: {samples}, Estimated Volume: {volume:.6f}")

def test_volume_conservation(pixel_spacing=(1, 1), slice_thickness=1):
    original_volume = monte_carlo_volume(lambda x, y, z: ellipsoid_blob(x, y, z, 1, 2, 3), 
                                         (0, 0, 0), (1, 2, 3), pixel_spacing, slice_thickness)
    
    # Test rotation
    rotated_volume = monte_carlo_volume(lambda x, y, z: rotated_blob(x, y, z, lambda x, y, z: ellipsoid_blob(x, y, z, 1, 2, 3), np.pi/4), 
                                        (0, 0, 0), (2, 2, 3), pixel_spacing, slice_thickness)
    
    # Test scaling
    scale_factor = 2
    scaled_volume = monte_carlo_volume(lambda x, y, z: scaled_blob(x, y, z, lambda x, y, z: ellipsoid_blob(x, y, z, 1, 2, 3), scale_factor), 
                                       (0, 0, 0), (2, 4, 6), pixel_spacing, slice_thickness)
    
    print(f"Original Volume: {original_volume:.6f}")
    print(f"Rotated Volume: {rotated_volume:.6f}, Difference: {abs(original_volume - rotated_volume) / original_volume * 100:.2f}%")
    print(f"Scaled Volume: {scaled_volume:.6f}, Expected: {original_volume * scale_factor**3:.6f}, Difference: {abs(scaled_volume - original_volume * scale_factor**3) / (original_volume * scale_factor**3) * 100:.2f}%")


def test_blob_volumes(pixel_spacing=(1, 1), slice_thickness=1):
    voxel_volume = pixel_spacing[0] * pixel_spacing[1] * slice_thickness

    # Test cube
    side_length = 2
    cube_volume = side_length**3 * voxel_volume
    mc_cube_volume = monte_carlo_volume(lambda x, y, z: cube_blob(x, y, z, side_length), 
                                        (0, 0, 0), (side_length/2, side_length/2, side_length/2),
                                        pixel_spacing, slice_thickness)
    
    # Test cylinder
    radius, height = 1, 2
    cylinder_volume = pi * radius**2 * height * voxel_volume
    mc_cylinder_volume = monte_carlo_volume(lambda x, y, z: cylinder_blob(x, y, z, radius, height), 
                                            (0, 0, 0), (radius, radius, height/2),
                                            pixel_spacing, slice_thickness)
    
    # Test ellipsoid
    a, b, c = 1, 2, 3
    ellipsoid_volume = 4/3 * pi * a * b * c * voxel_volume
    mc_ellipsoid_volume = monte_carlo_volume(lambda x, y, z: ellipsoid_blob(x, y, z, a, b, c), 
                                             (0, 0, 0), (a, b, c),
                                             pixel_spacing, slice_thickness)
    
    print(f"Cube - Actual: {cube_volume:.2f}, Monte Carlo: {mc_cube_volume:.2f}, Error: {abs(cube_volume - mc_cube_volume) / cube_volume * 100:.2f}%")
    print(f"Cylinder - Actual: {cylinder_volume:.2f}, Monte Carlo: {mc_cylinder_volume:.2f}, Error: {abs(cylinder_volume - mc_cylinder_volume) / cylinder_volume * 100:.2f}%")
    print(f"Ellipsoid - Actual: {ellipsoid_volume:.2f}, Monte Carlo: {mc_ellipsoid_volume:.2f}, Error: {abs(ellipsoid_volume - mc_ellipsoid_volume) / ellipsoid_volume * 100:.2f}%")

def test_all_monte_carlo():
    test_blob_volumes()
    test_blob_volumes((0.5, 0.5), 0.5)
    convergence_test(lambda x, y, z: ellipsoid_blob(x, y, z, 1, 2, 3), (0, 0, 0), (1, 2, 3))
    test_volume_conservation()
    