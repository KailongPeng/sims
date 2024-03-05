# import tkinter as tk
# import numpy as np
# import os

# class MatrixDrawer:
#     def __init__(self, master, rows=12, cols=12):
#         self.master = master
#         self.rows = rows
#         self.cols = cols
#         self.matrix1 = np.zeros((rows, cols), dtype=int)
#         self.matrix2 = np.zeros((rows, cols), dtype=int)
#         self.matrices_list = []  # List to hold all drawn matrices pairs
#         self.buttons1 = [[None for _ in range(cols)] for _ in range(rows)]
#         self.buttons2 = [[None for _ in range(cols)] for _ in range(rows)]
#
#         for i in range(rows):
#             for j in range(cols*2 + 1):  # Adjust to include space for the separator
#                 if j < cols:  # First matrix buttons
#                     button = tk.Button(master, bg="white", width=2, height=1,
#                                        command=lambda i=i, j=j: self.toggle_button(i, j, matrix='matrix1'))
#                     button.grid(row=i, column=j)
#                     self.buttons1[i][j] = button
#                 elif j == cols:  # Separator column
#                     separator = tk.Frame(master, bg="black", width=2, height=1)
#                     separator.grid(row=i, column=j, sticky="ns")
#                 else:  # Second matrix buttons
#                     button = tk.Button(master, bg="white", width=2, height=1,
#                                        command=lambda i=i, j=j-cols-1: self.toggle_button(i, j-cols-1, matrix='matrix2'))
#                     button.grid(row=i, column=j)
#                     self.buttons2[i][j-cols-1] = button
#
#         save_button = tk.Button(master, text="Save Matrices", command=self.save_matrices)
#         save_button.grid(row=rows, column=0, columnspan=cols, sticky="ew")
#
#         finish_button = tk.Button(master, text="Finish Drawing", command=self.finish_drawing)
#         finish_button.grid(row=rows, column=cols+1, columnspan=cols, sticky="ew")  # Adjust for separator
#
#     def toggle_button(self, i, j, matrix):
#         if matrix == 'matrix1':
#             self.matrix1[i, j] = 0 if self.matrix1[i, j] == 1 else 1
#             self.buttons1[i][j].configure(bg="lightgray" if self.matrix1[i, j] == 1 else "white")
#         else:  # matrix2
#             self.matrix2[i, j] = 0 if self.matrix2[i, j] == 1 else 1
#             self.buttons2[i][j].configure(bg="lightgray" if self.matrix2[i, j] == 1 else "white")
#
#     def save_matrices(self):
#         self.matrices_list.append((self.matrix1.copy(), self.matrix2.copy()))
#         self.matrix1 = np.zeros((self.rows, self.cols), dtype=int)
#         self.matrix2 = np.zeros((self.rows, self.cols), dtype=int)
#         for i in range(self.rows):
#             for j in range(self.cols):
#                 self.buttons1[i][j].configure(bg="white")
#                 self.buttons2[i][j].configure(bg="white")
#         print("Matrices saved and boards reset.")
#
#     def finish_drawing(self):
#         save_path = "./drawed"  # Update path as necessary
#         if not os.path.exists(save_path):
#             os.makedirs(save_path)
#         np.save(os.path.join(save_path, "matrices.npy"), np.array(self.matrices_list))
#         print(f"Finished drawing. Matrices saved to {os.path.join(save_path, 'matrices.npy')}")
#
# def main():
#     root = tk.Tk()
#     app = MatrixDrawer(root)
#     root.mainloop()

# if __name__ == "__main__":
#     main()


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def draw_line_on_canvas(center_position, angle, canvas_size=(12, 12), line_length=7, line_width=2):
    canvas = np.zeros(canvas_size)

    # 直线参数
    start_x, start_y = center_position
    angle_rad = np.radians(angle)

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

    bresenham_line(end_x1, end_y1, end_x2, end_y2)
    return canvas

def LGN_on_off(x, y, angle):
    canvas1 = draw_line_on_canvas(center_position=(x, y), angle=angle)
    canvas2 = draw_line_on_canvas(
        center_position=(round(x + 3 * np.cos(np.radians(90 - angle))), round(y - 3 * np.sin(np.radians(90 - angle)))),
        angle=angle)
    return canvas1, canvas2

# def generate_and_save_rectangles(num_calls):
#     rectangles = []
#     for _ in range(num_calls):
#         x, y = np.random.randint(1, 11, size=2)
#         angle = np.random.randint(0, 181)
#         canvas1, canvas2 = LGN_on_off(x, y, angle)
#         if np.any(canvas1) and np.any(canvas2):
#             rectangles.append((canvas1, canvas2))
#             # 可视化画布和直线
#             plt.figure()
#             plt.imshow(canvas1, cmap='Greys', interpolation='nearest')
#             plt.figure()
#             plt.imshow(canvas2, cmap='rainbow', interpolation='nearest')
#             plt.title(f'Line at ({x} {y}), Angle: {angle}°')
#             plt.show()
#
#     return rectangles

# rectangles_data = generate_and_save_rectangles(10)

def convert_to_list(canvas1, canvas2, index):
    """将canvas1和canvas2转换为列表"""
    flattened1 = canvas1.flatten()
    flattened2 = canvas2.flatten()
    return ['_D', f'X{index}'] + flattened1.tolist() + flattened2.tolist()

def generate_and_save_rectangles(num_calls, data_for_df):
    for i in range(1, num_calls + 1):  # 从1开始编号
        x, y = np.random.randint(1, 11, size=2)
        angle = np.random.randint(0, 181)
        canvas1, canvas2 = LGN_on_off(x, y, angle)
        if np.any(canvas1) and np.any(canvas2):
            row = convert_to_list(canvas1, canvas2, i)
            data_for_df.loc[len(data_for_df)] = row
    # # 创建DataFrame
    # cols = ['_H', '$Name'] + [f'%LGNon[2:{i // 12},{i % 12}]' for i in range(144)] + [f'%LGNoff[2:{i // 12},{i % 12}]' for i in range(144)]
    # probes_df = pd.DataFrame(data_for_df, columns=cols)
    # 保存到.tsv文件
    # probes_df.to_csv(file_path, sep='\t', index=False)
    return probes_df

# 调用函数生成数据并保存
file_path = '/gpfs/milgram/scratch60/turk-browne/kp578/chanales/v1rf/probes.tsv'
probes_df = pd.read_csv(file_path, sep='\t')
probes_df = generate_and_save_rectangles(10, probes_df)
probes_df.to_csv('/gpfs/milgram/scratch60/turk-browne/kp578/chanales/v1rf/probes_new.tsv',
                 sep='\t', index=False)


"""
对于产生的rectangles_data, 把每一对(canvas1, canvas2)都变成一个长度为290的一个list, 
其中第一个元素是'_D', 第二个元素是'Xn', 其中n是(canvas1, canvas2)的编号, 剩下的元素是canvas1和canvas2的flatten后的结果.
然后把这些list都append到一个probes_df里面, 然后把这个probes_df保存到一个.tsv文件里面.
其中probes_df 使用probes_df = pd.read_csv(file_path, sep='\t')的方式读取

"""