import pydicom
import numpy as np
import os

def load_patient_data(patient_folder):
    series_dict = {}
    for root, _, files in os.walk(patient_folder):
        for file in files:
            if file.endswith('.dcm'):
                ds = pydicom.dcmread(os.path.join(root, file))
                series_uid = ds.SeriesInstanceUID
                series_description = getattr(ds, 'SeriesDescription', 'No description')
                if series_uid not in series_dict:
                    series_dict[series_uid] = {'description': series_description, 'slices': []}
                series_dict[series_uid]['slices'].append(ds)
    
    # Sort each series by instance number
    for series_uid in series_dict:
        series_dict[series_uid]['slices'].sort(key=lambda x: x.InstanceNumber)
    
    return series_dict

def add_object_to_volume(volume, obj):
    # Convert both inputs to float for processing
    volume_float = volume.astype(float)
    obj_float = obj.astype(float)
    
    # Add the object with increased contrast
    result_float = np.where(obj_float > 0, volume_float + obj_float, volume_float)
    
    # Rescale to original range
    original_min, original_max = volume.min(), volume.max()
    result_scaled = (result_float - result_float.min()) / (result_float.max() - result_float.min())
    result_scaled = result_scaled * (original_max - original_min) + original_min
    
    # Clip and convert back to original data type
    result = np.clip(result_scaled, np.iinfo(volume.dtype).min, np.iinfo(volume.dtype).max).astype(volume.dtype)
    
    return result


