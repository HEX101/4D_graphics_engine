import ctypes

import numpy as np
import sdl2
import sdl2.ext
from numpy.typing import NDArray


class Renderer:
    def __init__(self, window: sdl2.SDL_Window, renderer: sdl2.SDL_Renderer) -> None:
        self.window = window
        self.renderer = renderer
        self.max_x = ctypes.c_int()
        self.max_y = ctypes.c_int()
        sdl2.SDL_GetWindowSize(window, self.max_x, self.max_y)

    def draw_edges(self, projected: NDArray[np.int_], edges: NDArray[np.int_]) -> None:
        sdl2.SDL_SetRenderDrawColor(self.renderer, 255, 255, 255, 255)
        for e in edges:
            if e[0] >= len(projected) or e[1] >= len(projected):
                continue

            v1 = projected[e[0]]
            v2 = projected[e[1]]

            self.draw_line(v1, v2)
                    
    def draw_line(self, v1: NDArray[np.int_], v2: NDArray[np.int_]) -> None:
        x1, y1 = v1[0], v1[1]
        x2, y2 = v2[0], v2[1]

        dx: int = abs(x2 - x1)
        dy: int = abs(y2 - y1)
        sx: int = 1 if x1 < x2 else -1
        sy: int = 1 if y1 < y2 else -1
        err: int = dx - dy

        while True:
            if 0 <= x1 < self.max_x.value - 1 and 0 <= y1 < self.max_y.value - 1:
                sdl2.SDL_RenderDrawPoint(self.renderer, x1, y1)

            if x1 == x2 and y1 == y2:
                break
            e2: int = err * 2
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy

    def clear(self) -> None:
        sdl2.SDL_SetRenderDrawColor(self.renderer, 0, 0, 0, 255)
        sdl2.SDL_RenderClear(self.renderer)

    def refresh(self) -> None:
        sdl2.SDL_RenderPresent(self.renderer)