from numba import njit, jit
import pygame as pg
import numpy as np
import time

SIZE = np.array((1080, 720))
step = 0.99
sensivity = 1
ztime = 0

mouse_pos = np.array((0, 0))
abs_mouse_pos = np.array((0, 0))
start_time = time.time()
surface = pg.display.set_mode(SIZE)
clock = pg.time.Clock()

X = np.arange(SIZE[0]*step, step=step)
Y = np.arange(SIZE[1]*step, step=step)

X, Y, = np.array(np.floor(np.meshgrid(Y, X)), dtype=np.int32)

# @njit(parallel=True, fastmath=True) 
@jit(nopython=True, fastmath=True, parallel=True, nogil=True)
def render(X, Y, oy, ox):
    x, y = X-ox, Y-oy

    # Z = x^y
    # Z = np.sin(X**2 + Y**2)
    Z = (x & y)**2
    # Z = np.sin(x**2 + y**2) * 255
    # Z = np.tan(x^y)
    # Z = x % y



    Z %= 256
    Z = np.expand_dims(Z, -1)
    return Z

def offset():
    return np.int16(mouse_pos * sensivity * step)
    dt = (time.time() - start_time)* 100
    rt = np.int8(dt)
    return (rt, ) * 2

def draw():
    Z = render(X, Y, *offset())
    # Z = np.array(Z, dtype=np.uint8)
    
    Z = np.repeat(Z, 3, axis=2)
    return Z

def zoom(evy):
    global X, Y, step
    step -= evy
    if step <= 0: step = 0.01
    elif step/min(SIZE) > 1733: step = 1733 * min(SIZE)
    step = round(step, 3)

    print('step: ', step)

    X = np.arange(SIZE[0]*step, step=step)#, dtype=np.int32
    Y = np.arange(SIZE[1]*step, step=step)#, dtype=np.int32
    
    X, Y = X[0:SIZE[0]], Y[0:SIZE[1]]

    X, Y, = np.array(np.floor(np.meshgrid(Y, X)), dtype=np.int32)

    

def handle_events(evnt: list[pg.event.Event]):
    global mouse_pos, abs_mouse_pos
    for ev in evnt:
        match ev.type:
            case pg.QUIT:
                pg.quit()
                quit()
            
            case pg.MOUSEMOTION:
                abs_mouse_pos = ev.pos
                mouse_pos = np.array(ev.pos) - SIZE/(2 * step)

            case pg.MOUSEWHEEL:
                mouse_pos = np.array(abs_mouse_pos) - SIZE/(2 * step)
                scale = ev.y / 10
                zoom(scale * step)



def handle_clock():
    clock.tick()
    fps = clock.get_fps()
    render_time = fps and 1000/fps
    pg.display.set_caption(f'FPS: {fps :.2f} | render {render_time:.1f}ms')

while True:
    handle_events(pg.event.get())

    handle_clock()
    # ztime = 1 / (start_time - time.time()) 
    # print(step)
    # zoom(ztime+step)

    
    pg.surfarray.blit_array(surface, draw())
    pg.display.flip()
