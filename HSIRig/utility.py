from datetime import datetime
PREFIX_NAME="QualiCrop"

def get_dataset_name():
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    return timestamp_str

