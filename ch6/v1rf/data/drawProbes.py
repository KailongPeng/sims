import tkinter as tk
import numpy as np
import os

class MatrixDrawer:
    def __init__(self, master, rows=12, cols=12):
        self.master = master
        self.rows = rows
        self.cols = cols
        self.matrix = np.zeros((rows, cols), dtype=int)
        self.matrices_list = []  # List to hold all drawn matrices
        self.buttons = [[None for _ in range(cols)] for _ in range(rows)]

        for i in range(rows):
            for j in range(cols):
                button = tk.Button(master, bg="white", width=2, height=1,
                                   command=lambda i=i, j=j: self.toggle_button(i, j))
                button.grid(row=i, column=j)
                self.buttons[i][j] = button

        save_button = tk.Button(master, text="Save Matrix", command=self.save_matrix)
        save_button.grid(row=rows, column=0, columnspan=cols//2, sticky="ew")

        finish_button = tk.Button(master, text="Finish Drawing", command=self.finish_drawing)
        finish_button.grid(row=rows, column=cols//2, columnspan=cols//2, sticky="ew")

    def toggle_button(self, i, j):
        self.matrix[i, j] = 0 if self.matrix[i, j] == 1 else 1
        self.buttons[i][j].configure(bg="lightgray" if self.matrix[i, j] == 1 else "white")

    def save_matrix(self):
        # Add the current matrix to the list and reset for a new drawing
        self.matrices_list.append(self.matrix.copy())
        self.matrix = np.zeros((self.rows, self.cols), dtype=int)
        for i in range(self.rows):
            for j in range(self.cols):
                self.buttons[i][j].configure(bg="white")
        print("Matrix saved and board reset.")

    def finish_drawing(self):
        # Save the list of matrices to a file
        save_path = "C:\GoGi\sims\ch6\\v1rf\data\drawed"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save(os.path.join(save_path, "matrix.npy"), np.array(self.matrices_list))
        print(f"Finished drawing. Matrices saved to {os.path.join(save_path, 'matrix.npy')}")

def main():
    root = tk.Tk()
    app = MatrixDrawer(root)
    root.mainloop()

if __name__ == "__main__":
    main()
