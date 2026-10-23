import numpy as np
from pathlib import Path
from numpy.typing import NDArray

def test_load():
    from PIL import Image
    # create a dummy image with padding
    img = np.zeros((100, 100), dtype=np.uint8)
    img[20:80, 20:80] = np.random.randint(50, 200, (60, 60))
    Image.fromarray(img).save("test_pad.png")

test_load()
