import glfw
from OpenGL.GL import glBegin, glEnd, glColor3f, glVertex2f, glClear, glOrtho, GL_POINTS, GL_COLOR_BUFFER_BIT

def initialize_window(width: int, height: int) -> glfw._GLFWwindow | None:
    if not glfw.init():
        return None
    window = glfw.create_window(width, height, "Pixel Drawer", None, None)
    if not window:
        glfw.terminate()
        return None
    glfw.make_context_current(window)
    return window

def draw_pixel(x: float, y: float, r: float, g: float, b: float) -> None:
    glBegin(GL_POINTS)
    glColor3f(r, g, b)
    glVertex2f(x, y)
    glEnd()

def main() -> None:
    width, height = 800, 600
    window = initialize_window(width, height)
    if not window:
        return

    glOrtho(0, width, 0, height, -1, 1)

    while not glfw.window_should_close(window):
        glClear(GL_COLOR_BUFFER_BIT)
        
        draw_pixel(100, 100, 1.0, 0.0, 0.0)
        draw_pixel(200, 200, 0.0, 1.0, 0.0)
        draw_pixel(300, 300, 0.0, 0.0, 1.0)

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()

if __name__ == "__main__":
    main()
