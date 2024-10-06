import math
import sys
import time
from typing import Dict, Tuple

import numpy as np
import sdl2
from numpy.typing import NDArray

from renderer import Renderer
from shape_def import Shape

FRAMES_PER_SECOND = 60

class Projection:
    def __init__(self, max_x: int, max_y: int) -> None:
        self.max_x = max_x
        self.max_y = max_y
        
    def project_vertices(self, vertices: NDArray[np.float64], position: NDArray[np.float64]) -> NDArray[np.int_]:
        """
        Projects N-dimensional vertices onto a 2D plane.

        Args:
            vertices (NDArray[np.float64]): Array of N-dimensional vertex coordinates (n, N).
            position (NDArray[np.float64]): N-dimensional reference position (N,).

        Returns:
            NDArray[np.int_]: Array of projected 2D coordinates (n, 2).
        """
        dimension_to_project_to: int = 2

        num_vertices = vertices.shape[0]
        projected_vertices = np.zeros((num_vertices, dimension_to_project_to))

        half_max_x: float = self.max_x / 2
        half_max_y: float = self.max_y / 2

        diff_array = position[dimension_to_project_to:] - vertices[:, dimension_to_project_to:]

        epsilon = 1e-10
        scale_factors = np.prod(1 / (diff_array + epsilon), axis=1)

        vertex_part = vertices[:, :dimension_to_project_to]

        projected_vertices[:, 0] = half_max_x + (vertex_part[:, 0] * half_max_x * scale_factors)
        projected_vertices[:, 1] = half_max_y - (vertex_part[:, 1] * half_max_y * scale_factors)

        return projected_vertices.astype(np.int_)

class Rotation:
    def __init__(self, n: int) -> None:
        """
        Initialize Rotation for N dimensions.

        Args:
            n (int): Number of dimensions
        """
        self.n = n
        self.rotation_angles: Dict[Tuple[int, int], float] = {}
        self.rotation_speed: float = math.pi / 16
        self.has_changed: bool = True

    def rotate_camera(self, plane: tuple[int, int], delta_angle: float) -> None:
        """
        Rotate the camera around a specified plane.

        Args:
            plane (tuple[int, int]): ple of two axes indices defining the rotation plane.
            delta_angle (float): Incremental angle to rotate.
        """
        if plane not in self.rotation_angles:
            self.rotation_angles[plane] = 0.0
        self.rotation_angles[plane] += delta_angle * self.rotation_speed
        self.has_changed = True

    def get_rotation_matrix(self) -> NDArray[np.float64]:
        """
        Compute the cumulative rotation matrix for all specified planes.

        Returns:
            NDArray[np.float64]: N x N rotation matrix.
        """
        rotation_matrix = np.identity(self.n)
        for plane, angle in self.rotation_angles.items():
            i, j = plane
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            rotation = np.identity(self.n)
            rotation[i, i] = cos_a
            rotation[i, j] = -sin_a
            rotation[j, i] = sin_a
            rotation[j, j] = cos_a
            rotation_matrix = rotation_matrix @ rotation
        return rotation_matrix

    def rotate(self, vertices: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Apply all rotations to the vertices.

        Args:
            vertices (NDArray[np.float64]): Array of vertices with shape (num_vertices, N).

        Returns:
            NDArray[np.float64]: Rotated vertices.
        """
        rotation_matrix = self.get_rotation_matrix()
        return vertices @ rotation_matrix
    
class Position:
    def __init__(self, n:int) -> None:
        self.coords: NDArray[np.float64] = np.zeros((n,), dtype=np.float64)
        self.coords[2] = 3.0
        self.movement_speed: float = 0.1
        self.has_changed: bool = True
                
    def move(self, direction_vector: NDArray[np.float64]) -> None:
        self.coords += direction_vector * self.movement_speed
        self.has_changed = True
        
class InputHandler:
    def __init__(self, position: Position, rotation: Rotation) -> None:
        self.position = position
        self.rotation = rotation
        self.key_actions = {
            sdl2.SDLK_s: lambda: self.rotate_plane((1, 2),  1),
            sdl2.SDLK_w: lambda: self.rotate_plane((1, 2), -1),

            sdl2.SDLK_a: lambda: self.rotate_plane((0, 2),  1),
            sdl2.SDLK_d: lambda: self.rotate_plane((0, 2), -1),

            sdl2.SDLK_e: lambda: self.rotate_plane((0, 1),  1),
            sdl2.SDLK_q: lambda: self.rotate_plane((0, 1), -1),
            
            sdl2.SDLK_x: self.move_forward,
            sdl2.SDLK_z: self.move_backward,
        }
        self.keys_pressed: set[int] = set()

    def handle_input(self, event: sdl2.SDL_Event) -> None:
        if event.type == sdl2.SDL_QUIT:
            self.quit()

        if event.type == sdl2.SDL_KEYDOWN:
            self.keys_pressed.add(event.key.keysym.sym)
            self.perform_actions()

        if event.type == sdl2.SDL_KEYUP:
            self.keys_pressed.discard(event.key.keysym.sym)

    def perform_actions(self) -> None:
        for key in self.keys_pressed:
            if key in self.key_actions:
                self.key_actions[key]()

    def rotate_plane(self, plane: tuple[int, int], direction: int) -> None:
        """
        Rotate around a specified plane.
        
        Args:
            plane (tuple[int, int]): Tuple of two axis indices defining the rotation plane (e.g., (0, 1) for XY plane).
            direction (int): Rotation direction. Use 1 for positive rotation and -1 for negative rotation.
        """
        self.rotation.rotate_camera(plane=plane, delta_angle=direction)

    def move_forward(self) -> None:
        self.position.move(np.array([0.0, 0.0, -1.0]))

    def move_backward(self) -> None:
        self.position.move(np.array([0.0, 0.0, 1.0]))

    def quit(self) -> None:
        sdl2.SDL_Quit()
        sys.exit()

class Application:
    def __init__(self, n: int) -> None:
        """
        Initialize the application. 

        Args:
            n (int): Number of dimensions.
        """
        self.shape: Shape = Shape.define_n_dimensional_cube(n)
        self.position: Position = Position(n)
        self.rotation: Rotation = Rotation(n)
        
        sdl2.SDL_Init(sdl2.SDL_INIT_VIDEO)
        self.window = sdl2.SDL_CreateWindow(b"3D Projection", sdl2.SDL_WINDOWPOS_UNDEFINED, sdl2.SDL_WINDOWPOS_UNDEFINED, 800, 600, sdl2.SDL_WINDOW_SHOWN)
        renderer = sdl2.SDL_CreateRenderer(self.window, -1, sdl2.SDL_RENDERER_ACCELERATED)
        self.renderer: Renderer = Renderer(self.window, renderer)
        self.projection = Projection(800, 600)
        self.input_handler = InputHandler(self.position, self.rotation)

    def run(self) -> None:
        frame_duration = 1 / FRAMES_PER_SECOND
        while True:
            start_time = time.time()
            self.clear_and_render()
            self.handle_events()
            elapsed_time = time.time() - start_time
            time.sleep(max(0, frame_duration - elapsed_time))

    def clear_and_render(self) -> None:
        self.renderer.clear()
        if self.position.has_changed or self.rotation.has_changed:
            self.update_rendering()
            self.renderer.refresh()

    def update_rendering(self) -> None:
        rotated = self.rotation.rotate(self.shape.vertices)
        rotated[:, :2] -= self.position.coords[:2]
        projected = self.projection.project_vertices(rotated, self.position.coords)
        self.renderer.draw_edges(projected, self.shape.edges)
        self.position.has_changed, self.rotation.has_changed = False, False

    def handle_events(self) -> None:
        while True:
            event = sdl2.SDL_Event()
            if not sdl2.SDL_PollEvent(event):
                break
            self.input_handler.handle_input(event)
            
def main() -> None:
    app = Application(n=3)
    app.run()
    sdl2.SDL_DestroyRenderer(app.renderer)
    sdl2.SDL_DestroyWindow(app.window)
    sdl2.SDL_Quit()

if __name__ == "__main__":
    main()