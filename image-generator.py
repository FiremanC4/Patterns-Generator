from numba import njit, jit
from PIL import Image 
import numpy as np
import time

size = 1024 * 25

RAM = ((size**2) * 48) / 8388608
MSG = f'Picture size: {size}x{size}\n{RAM}MB + ≈500MB of RAM will be used. \nPress Y to continue: '
while input(MSG) not in ('y', 'Y'): pass

SIZE = np.array((size,) * 2)
step = 1


X = np.arange(SIZE[0]*step, step=step, dtype=np.uint16)
Y = np.arange(SIZE[1]*step, step=step, dtype=np.uint16)

X, Y, = np.meshgrid(Y, X)

@jit(nopython=True, fastmath=True, parallel=True, nogil=True)
def render():
    # Z = x^y
    # Z = np.sin(X**2 + Y**2)
    Z = (X & Y)**2
    # Z = np.sin(x**2 + y**2) * 255
    # Z = np.tan(X^Y)
    # Z = X % Y

    Z %= 256
    Z = np.expand_dims(Z, -1)
    return Z

def draw():
    Z = render()
    Z = np.uint8(Z)
    
    Z = np.repeat(Z, 3, axis=2)
    return Z


t1 = time.time()
R = draw()
print('Rendering time: ', time.time() - t1)

img = Image.fromarray(R)

# img.show()
img.save(r'C:\FiremanC4\python\tests\pillow-tests\IMG.png')



