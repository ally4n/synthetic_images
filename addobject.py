from math import pi
from object_utils import *
from dicom_utils import *
from test_utils import *

def process_series(series_list, object_type='sphere', object_size=10, position=(100, 100, 50), max_boundaries=None):

    # Get metadata from first slice
    first_slice = series_list[0]
    pixel_spacing = first_slice.PixelSpacing
    slice_thickness = first_slice.SliceThickness
    
    # Determine volume shape
    rows, cols = first_slice.pixel_array.shape
    num_slices = len(series_list)
    volume_shape = (num_slices, rows, cols)
    
    print(f"Original volume shape: {volume_shape}")
    
    # Create object
    if object_type == 'sphere':
        obj = create_sphere(object_size / 2, position, volume_shape, pixel_spacing, slice_thickness)
        # Calculate theoretical sphere volume
        radius = object_size / 2
        object_volume = (4/3) * pi * (radius ** 3)
    elif object_type == 'blob':
        if max_boundaries is None:
            raise ValueError("max_boundaries must be provided for blob objects")
        obj, blob_function = create_random_blob(position, max_boundaries, volume_shape, pixel_spacing, slice_thickness)
        print(f"Blob created with shape: {obj.shape}")
        print(f"Blob min value: {obj.min()}, max value: {obj.max()}")
        # Calculate blob volume using Monte Carlo method
        object_volume = monte_carlo_volume(blob_function, position, max_boundaries, pixel_spacing, slice_thickness)
    else:
        raise ValueError("Unsupported object type. Only 'sphere' and 'blob' are currently supported.")

    # Process each slice individually
    new_series = []
    for i, original_slice in enumerate(series_list):
        # Get pixel array
        pixel_array = original_slice.pixel_array
        
        # Check dimensions
        if pixel_array.shape != (rows, cols):
            print(f"Warning: Slice {i} has different dimensions: {pixel_array.shape}")
        
        # Add object to this slice
        modified_slice = add_object_to_volume(pixel_array, obj[i])
        
        print(f"Slice {i}: Original range: [{pixel_array.min()}, {pixel_array.max()}], "
              f"Modified range: [{modified_slice.min()}, {modified_slice.max()}]")
        
        # Create a new DICOM object
        new_slice = pydicom.Dataset()
        
        # Copy all elements from the original slice
        for elem in original_slice:
            new_slice.add(elem)
        
        # Update pixel data
        new_slice.PixelData = modified_slice.tobytes()
        
        # Preserve or update important DICOM tags
        for tag in ['WindowCenter', 'WindowWidth', 'RescaleSlope', 'RescaleIntercept', 
                    'BitsAllocated', 'BitsStored', 'HighBit', 'PixelRepresentation']:
            if tag in original_slice:
                setattr(new_slice, tag, getattr(original_slice, tag))
        
        # Ensure file_meta is preserved
        if hasattr(original_slice, 'file_meta'):
            new_slice.file_meta = original_slice.file_meta
        
        # Update image type to indicate processing
        if 'ImageType' in new_slice:
            new_slice.ImageType = list(new_slice.ImageType) + ['PROCESSED']
        
        # Ensure Rows and Columns are correct
        new_slice.Rows, new_slice.Columns = modified_slice.shape
        
        new_series.append(new_slice)
        
        #print(f"Processed slice {i}: Original shape {pixel_array.shape}, Modified shape {modified_slice.shape}")
        #print(f"Original range: [{pixel_array.min()}, {pixel_array.max()}], Modified range: [{modified_slice.min()}, {modified_slice.max()}]")
    
    return new_series, object_volume

def main():
    root_folder = os.getcwd()  # Current working directory
    data_folder = os.path.join(root_folder, "data")
    output_folder = os.path.join(root_folder, "output")
    
    # List all patient folders
    patient_folders = [f for f in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, f))]
    
    print("Available patients:")
    for i, patient in enumerate(patient_folders):
        print(f"{i+1}. {patient}")
    
    patient_choice = int(input("Enter the number of the patient to process: ")) - 1
    selected_patient = patient_folders[patient_choice]
    patient_folder = os.path.join(data_folder, selected_patient)
    
    # Load all series for the patient
    series_dict = load_patient_data(patient_folder)
    
    # Display available series
    print(f"\nAvailable series for {selected_patient}:")
    for i, (series_uid, series_info) in enumerate(series_dict.items()):
        print(f"{i+1}. {series_info['description']} (UID: {series_uid}, Slices: {len(series_info['slices'])})")
    
    # Let user choose which series to process
    series_choice = int(input("Enter the number of the series to process: ")) - 1
    selected_series_uid = list(series_dict.keys())[series_choice]
    
    # Get the selected series
    selected_series = series_dict[selected_series_uid]['slices']
    
    # Extract volume dimensions
    num_slices = len(selected_series)
    rows, cols = selected_series[0].pixel_array.shape
    
    print(f"Selected series dimensions: {cols}x{rows}x{num_slices}")

    # Get object type from user
    object_type = input("Enter the type of object to add (sphere or blob): ").lower()
    while object_type not in ['sphere', 'blob']:
        print("Invalid object type. Please enter 'sphere' or 'blob'.")
        object_type = input("Enter the type of object to add (sphere or blob): ").lower()
    
    # Get object parameters from user
    if object_type == 'sphere':
        object_size = float(input("Enter the diameter of the sphere in mm: "))
    else:  # blob
        object_size = float(input("Enter the maximum size of the blob in mm (try a larger value, e.g., 50): "))
    
    z = float(input(f"Enter the x-coordinate of the object center (try middle, e.g., {cols//2}): "))
    y = float(input(f"Enter the y-coordinate of the object center (try middle, e.g., {rows//2}): "))
    x = float(input(f"Enter the z-coordinate of the object center (try middle, e.g., {num_slices//2}): "))
    
    # Process the selected series
    if object_type == 'sphere':
        new_series, object_volume = process_series(selected_series, 
                                                   object_type='sphere', 
                                                   object_size=object_size, 
                                                   position=(x, y, z))
    else:  # blob
        max_boundaries = (object_size/2, object_size/2, object_size/2)
        new_series, object_volume = process_series(selected_series, 
                                                   object_type='blob', 
                                                   object_size=object_size, 
                                                   position=(x, y, z),
                                                   max_boundaries=max_boundaries)
    
    # Save the processed series
    patient_output_folder = os.path.join(output_folder, selected_patient)
    os.makedirs(patient_output_folder, exist_ok=True)
    for i, slice in enumerate(new_series):
        slice.save_as(os.path.join(patient_output_folder, f'processed_slice_{i:03d}.dcm'))
    
    print(f"\nProcessed series saved to {patient_output_folder}")
    print(f"Added object volume: {object_volume:.2f} mm³")
   
if __name__ == "__main__":
    #test_all_monte_carlo()
    main()