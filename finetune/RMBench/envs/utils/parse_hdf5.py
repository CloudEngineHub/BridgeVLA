import h5py, cv2
import numpy as np


def parse_img_array(data):
    """
    Decode an array of byte streams into an array of images.

    Args:
        data: np.ndarray of shape (N,); each element is either Python bytes or an np.ndarray(dtype=uint8)
    Returns:
        imgs: np.ndarray of shape (N, H, W, C), dtype=uint8
    """
    # Make sure data is an iterable 1-D array
    flat = data.ravel()

    imgs = []
    for buf in flat:
        # buf may be bytes or np.ndarray(dtype=uint8)
        if isinstance(buf, (bytes, bytearray)):
            arr = np.frombuffer(buf, dtype=np.uint8)
        elif isinstance(buf, np.ndarray) and buf.dtype == np.uint8:
            arr = buf
        else:
            raise TypeError(f"Unsupported buffer type: {type(buf)}")

        # Decode into a BGR image
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2.imdecode returned None: the byte stream is probably not a valid image format")
        imgs.append(img)

    # Turn the list into an ndarray of shape (N, H, W, C)
    return np.stack(imgs, axis=0)


def h5_to_dict(node):
    result = {}
    for name, item in node.items():
        if isinstance(item, h5py.Dataset):
            data = item[()]
            if "rgb" in name:
                result[name] = parse_img_array(data)
            else:
                result[name] = data
        elif isinstance(item, h5py.Group):
            # Recurse into the sub-group
            result[name] = h5_to_dict(item)
    # To read the attributes as well, you could:
    if hasattr(node, "attrs") and len(node.attrs) > 0:
        result["_attrs"] = dict(node.attrs)
    return result


def read_hdf5(file_path):
    with h5py.File(file_path, "r") as f:
        data_dict = h5_to_dict(f)
    return data_dict
