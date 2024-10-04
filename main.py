import curses
import itertools
import math
from typing import Callable, Dict, Optional

import numpy as np
from numpy.typing import NDArray


class Projection:
    def __init__(self, max_x: int, max_y: int) -> None:
        self.max_x = max_x
        self.max_y = max_y
        
    def project_vertices(self, vertices: NDArray[np.float64], position: NDArray[np.float64]) -> NDArray[np.int_]:
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

class Renderer:
    def __init__(self, stdscr: curses.window) -> None:
        self.stdscr = stdscr
        self.max_y, self.max_x = stdscr.getmaxyx()
        curses.curs_set(0)
        curses.start_color()
        curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLACK)
        self.stdscr.nodelay(True)
        self.stdscr.keypad(True)

    def draw_edges(self, projected: NDArray[np.int_], edges: NDArray[np.int_]) -> None:
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
            if 0 <= x1 < self.max_x - 1 and 0 <= y1 < self.max_y - 1:
                self.stdscr.addch(y1, x1, ".", curses.color_pair(1))

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
        self.stdscr.clear()

    def refresh(self) -> None:
        self.stdscr.refresh()

class Shape:
    def __init__(self, vertices: NDArray[np.float64], edges: NDArray[np.int_]) -> None:
        self.vertices: NDArray[np.float64] = vertices
        self.edges: NDArray[np.int_] = edges
            
    @staticmethod
    def define_n_dimensional_cube(n: int) -> 'Shape':        
        vertices = np.array(list(itertools.product([-1, 1], repeat=n)), dtype=np.float64)

        edges = []
        dimension = vertices.shape[1]
        
        vertex_to_index = {tuple(vertex): idx for idx, vertex in enumerate(vertices)}
        
        for idx, vertex in enumerate(vertices):
            for dim in range(dimension):
                neighbor = vertex.copy()
                neighbor[dim] *= -1
                neighbor_tuple = tuple(neighbor)
                neighbor_idx = vertex_to_index.get(neighbor_tuple)
                if neighbor_idx is not None and neighbor_idx > idx:
                    edges.append((idx, neighbor_idx))
        
        edges_array = np.array(edges, dtype=np.int_)

        return Shape(vertices, edges_array)

class Rotation: # generlize to apply to n dimensions
    def __init__(self, yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0) -> None:
        self.yaw = yaw                  # Rotation around YZ plane
        self.pitch = pitch              # Rotation around XZ plane
        self.roll = roll                # Rotation around XY plane
        self.rotation_speed: float = math.pi / 16
        self.has_changed: bool = True

    def rotate_camera(self, dtheta: float = 0.0, dphi: float = 0.0, dw: float = 0.0) -> None:
        self.yaw += dtheta * self.rotation_speed
        self.pitch += dphi * self.rotation_speed
        self.has_changed = True 

    def rotate_y(self, vertices: NDArray[np.float64]) -> NDArray[np.float64]:
        sin_theta: float = math.sin(self.yaw)
        cos_theta: float = math.cos(self.yaw)
        roty: NDArray[np.float64] = np.array([[cos_theta, 0, -sin_theta],
                                             [0, 1, 0],
                                             [sin_theta, 0, cos_theta]])
        rotated_vertices: NDArray[np.float64] = np.dot(vertices, roty)
        return rotated_vertices

    def rotate_x(self, vertices: NDArray[np.float64]) -> NDArray[np.float64]:
        sin_phi: float = math.sin(self.pitch)
        cos_phi: float = math.cos(self.pitch)
        rotx: NDArray[np.float64] = np.array([[1, 0, 0],
                                             [0, cos_phi, -sin_phi],
                                             [0, sin_phi, cos_phi]])
        rotated_vertices: NDArray[np.float64] = np.dot(vertices, rotx)
        return rotated_vertices
    
    def rotate_z(self, vertices: NDArray[np.float64]) -> NDArray[np.float64]:
        sin_psi: float = math.sin(self.roll)
        cos_psi: float = math.cos(self.roll)
        
        rotz: NDArray[np.float64] = np.array([[cos_psi, -sin_psi, 0],
                                            [sin_psi, cos_psi, 0],
                                            [0, 0, 1, 0]])
        rotated_vertices: NDArray[np.float64] = np.dot(vertices, rotz)
        return rotated_vertices

    def rotate(self, vertices: NDArray[np.float64]) -> NDArray[np.float64]:
        rotated = self.rotate_y(vertices)
        rotated = self.rotate_x(rotated)
        return rotated
    
class Position:
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        self.coords: NDArray[np.float64] = np.array([x, y, z])
        self.movement_speed: float = 0.1
        self.has_changed: bool = True
        
    def move(self, direction_vector: NDArray[np.float64]) -> None:
        self.coords += direction_vector * self.movement_speed
        self.has_changed = True

class InputHandler:         # generlize for n dimensions
    def __init__(self, position: Position, rotation: Rotation) -> None:
        self.position = position
        self.rotation = rotation
        self.key_actions: Dict[int, Callable[[], None]] = {
            ord('w'): lambda: self.rotation.rotate_camera(dphi=-1),  # Rotate up (pitch decrease)
            ord('s'): lambda: self.rotation.rotate_camera(dphi=1),   # Rotate down (pitch increase)
            ord('a'): lambda: self.rotation.rotate_camera(dtheta=1), # Rotate left (yaw increase)
            ord('d'): lambda: self.rotation.rotate_camera(dtheta=-1),# Rotate right (yaw decrease)
            ord('z'): lambda: self.position.move(np.array([0.0, 0.0, -1.0])),  # Move forward (Z axis)
            ord('x'): lambda: self.position.move(np.array([0.0, 0.0, 1.0])),   # Move backward (Z axis)
        }

    def handle_input(self, key: int) -> None:
        action = self.key_actions.get(key)
        if action:
            action()

class Application:
    def __init__(self, n: int) -> None:
        self.shape: Shape = Shape.define_n_dimensional_cube(n)
        self.position: Position = Position(z=3)
        self.rotation: Rotation = Rotation()
        self.renderer: Optional[Renderer] = None
        self.projection: Optional[Projection] = None
        self.input_handler: Optional[InputHandler] = None

    def run(self, stdscr: curses.window) -> None:
        self.renderer = Renderer(stdscr)
        self.projection = Projection(self.renderer.max_x, self.renderer.max_y)
        self.input_handler = InputHandler(self.position, self. rotation)

        while True:
            if self.position.has_changed or self.rotation.has_changed:
                self.renderer.clear()
                
                stdscr.addstr(0, 0, f'X: {self.position.coords[0]:.2f}, Z: {self.position.coords[2]:.2f}', curses.color_pair(1))
                stdscr.addstr(1, 0, f'Yaw: {round((self.rotation.yaw * (180 / math.pi) % 360), 2)}, Pitch: {round((self.rotation.pitch * (180 / math.pi) % 360), 2)}', curses.color_pair(1))
                
                rotated: NDArray[np.float64] = self.rotation.rotate(self.shape.vertices)
                rotated[:, :2] -= self.position.coords[:2]
                projected: NDArray[np.int_] = self.projection.project_vertices(
                    rotated, self.position.coords
                )

                self.renderer.draw_edges(projected, self.shape.edges)
                self.position.has_changed, self.rotation.has_changed = False, False
                

                self.renderer.refresh()

            key: int = stdscr.getch()
            if key == ord('n'):
                break
            self.input_handler.handle_input(key)
            
def main() -> None:
    app = Application(n=3)
    curses.wrapper(app.run)

if __name__ == "__main__":
    main()