import tkinter as tk
import numpy as np
import os

class MatrixDrawer:
    def __init__(self, master, rows=12, cols=12):
        self.master = master
        self.rows = rows
        self.cols = cols
        self.matrix1 = np.zeros((rows, cols), dtype=int)
        self.matrix2 = np.zeros((rows, cols), dtype=int)
        self.matrices_list = []  # List to hold all drawn matrices pairs
        self.buttons1 = [[None for _ in range(cols)] for _ in range(rows)]
        self.buttons2 = [[None for _ in range(cols)] for _ in range(rows)]

        for i in range(rows):
            for j in range(cols*2 + 1):  # Adjust to include space for the separator
                if j < cols:  # First matrix buttons
                    button = tk.Button(master, bg="white", width=2, height=1,
                                       command=lambda i=i, j=j: self.toggle_button(i, j, matrix='matrix1'))
                    button.grid(row=i, column=j)
                    self.buttons1[i][j] = button
                elif j == cols:  # Separator column
                    separator = tk.Frame(master, bg="black", width=2, height=1)
                    separator.grid(row=i, column=j, sticky="ns")
                else:  # Second matrix buttons
                    button = tk.Button(master, bg="white", width=2, height=1,
                                       command=lambda i=i, j=j-cols-1: self.toggle_button(i, j-cols-1, matrix='matrix2'))
                    button.grid(row=i, column=j)
                    self.buttons2[i][j-cols-1] = button

        save_button = tk.Button(master, text="Save Matrices", command=self.save_matrices)
        save_button.grid(row=rows, column=0, columnspan=cols, sticky="ew")

        finish_button = tk.Button(master, text="Finish Drawing", command=self.finish_drawing)
        finish_button.grid(row=rows, column=cols+1, columnspan=cols, sticky="ew")  # Adjust for separator

    def toggle_button(self, i, j, matrix):
        if matrix == 'matrix1':
            self.matrix1[i, j] = 0 if self.matrix1[i, j] == 1 else 1
            self.buttons1[i][j].configure(bg="lightgray" if self.matrix1[i, j] == 1 else "white")
        else:  # matrix2
            self.matrix2[i, j] = 0 if self.matrix2[i, j] == 1 else 1
            self.buttons2[i][j].configure(bg="lightgray" if self.matrix2[i, j] == 1 else "white")

    def save_matrices(self):
        self.matrices_list.append((self.matrix1.copy(), self.matrix2.copy()))
        self.matrix1 = np.zeros((self.rows, self.cols), dtype=int)
        self.matrix2 = np.zeros((self.rows, self.cols), dtype=int)
        for i in range(self.rows):
            for j in range(self.cols):
                self.buttons1[i][j].configure(bg="white")
                self.buttons2[i][j].configure(bg="white")
        print("Matrices saved and boards reset.")

    def finish_drawing(self):
        save_path = "./drawed"  # Update path as necessary
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save(os.path.join(save_path, "matrices.npy"), np.array(self.matrices_list))
        print(f"Finished drawing. Matrices saved to {os.path.join(save_path, 'matrices.npy')}")

def main():
    root = tk.Tk()
    app = MatrixDrawer(root)
    root.mainloop()

if __name__ == "__main__":
    main()

import numpy as np
import matplotlib.pyplot as plt


def draw_line_on_canvas(center_position, angle, canvas_size=(12, 12), line_length=7, line_width=2):
    canvas = np.zeros(canvas_size)  # 初始化画布

    # 直线参数
    start_x, start_y = center_position  # 直线中心位置
    angle_rad = np.radians(angle)  # 将角度转换为弧度

    # 计算直线的两个终点坐标
    half_length = line_length / 2
    end_x1 = int(start_x + half_length * np.cos(angle_rad))
    end_y1 = int(start_y + half_length * np.sin(angle_rad))
    end_x2 = int(start_x - half_length * np.cos(angle_rad))
    end_y2 = int(start_y - half_length * np.sin(angle_rad))

    # 通过Bresenham算法画直线
    def bresenham_line(x0, y0, x1, y1):
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while True:
            # 对直线的每个点进行宽度调整
            if 0 <= x0 < canvas_size[0] and 0 <= y0 < canvas_size[1]:
                for wx in range(-line_width + 1, line_width):
                    for wy in range(-line_width + 1, line_width):
                        if 0 <= x0 + wx < canvas_size[0] and 0 <= y0 + wy < canvas_size[1]:
                            canvas[y0 + wy, x0 + wx] = 1
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    # 画直线
    bresenham_line(end_x1, end_y1, end_x2, end_y2)

    # 可视化画布和直线
    plt.imshow(canvas, cmap='Greys', interpolation='nearest')
    plt.title(f'Line at ({center_position}), Angle: {angle}°')
    plt.show()


# 调用函数示例
def LGN_on_off(x, y, angle):
    # 角度转换为弧度进行计算
    angle_rad = np.radians(angle)
    draw_line_on_canvas(center_position=(x, y), angle=angle)
    draw_line_on_canvas(
        center_position=(round(x + 3 * np.cos(np.radians(90 - angle))), round(y - 3 * np.sin(np.radians(90 - angle)))),
        angle=angle)


LGN_on_off(5, 5, 0)

# 对于以上代码, 每一次调用都可以生成两个包含0或者1的矩形. 我希望可以保存下来这些矩形. 与此同时, 我希望可以多次随机调用这个函数, 并且将生成的矩形保存下来. 请问如何实现这个功能?