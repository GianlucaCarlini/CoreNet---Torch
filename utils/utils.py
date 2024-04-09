import numpy as np


def view_array_patches(img, window_size, strides, padding="valid") -> np.ndarray:
    """Generates a view of the patches of an array.

    Parameters
    ----------
    img: np.ndarray
        The input array to be patched.
    window_size: int
        The size of the window for the patches.
    strides: int
        The stride of the sliding window.
    padding: str, optional
        The type of padding to apply to the input image.
        It is compliant with the tensorflow definitions of padding.
        Defaults to 'valid'.

    Returns
    -------
    np.ndarray
        A view of the patches of the input array.

    """

    h, w = img.shape[:2]

    if padding == "same":
        out_height = np.ceil(h / strides[0]).astype(np.uint8)
        out_width = np.ceil(w / strides[1]).astype(np.uint8)

        if h % strides[0] == 0:
            pad_along_height = max(window_size[0] - strides[0], 0)
        else:
            pad_along_height = max(window_size[0] - (h % strides[0]), 0)
        if w % strides[1] == 0:
            pad_along_width = max(window_size[1] - strides[1], 0)
        else:
            pad_along_width = max(window_size[1] - (w % strides[1]), 0)

        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left

        if img.ndim > 2:
            x = np.pad(
                img, pad_width=((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
            )
        else:
            x = np.pad(img, pad_width=((pad_top, pad_bottom), (pad_left, pad_right)))

    elif padding == "valid":
        out_height = np.ceil((h - window_size[0] + 1) / strides[0]).astype(np.uint8)
        out_width = np.ceil((w - window_size[1] + 1) / strides[1]).astype(np.uint8)

        x = img[
            : window_size[0] + out_height * strides[0],
            : window_size[1] + out_width * strides[1],
        ]

    else:
        raise NotImplementedError("Padding type not implemented.")

    stride0, stride1 = x.strides[:2]
    Wh, Ww = window_size
    stride_w, stride_h = strides

    view_shape = (out_height, out_width) + (Wh, Ww) + x.shape[2:]
    strides_ = (
        (stride_h * stride0, stride_w * stride1) + (stride0, stride1) + x.strides[2:]
    )

    view = np.lib.stride_tricks.as_strided(x, view_shape, strides=strides_)

    return view
