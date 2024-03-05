import tkinter as tk
import numpy as np

class MatrixDrawer:
    def __init__(self, master, rows=12, cols=12):
        self.master = master
        self.rows = rows
        self.cols = cols
        self.matrix = np.zeros((rows, cols), dtype=int)
        self.buttons = [[None for _ in range(cols)] for _ in range(rows)]

        for i in range(rows):
            for j in range(cols):
                button = tk.Button(master, bg="white", width=2, height=1,
                                   command=lambda i=i, j=j: self.toggle_button(i, j))
                button.grid(row=i, column=j)
                self.buttons[i][j] = button

        save_button = tk.Button(master, text="Save Matrix", command=self.save_matrix)
        save_button.grid(row=rows, column=0, columnspan=cols, sticky="ew")

    def toggle_button(self, i, j):
        self.matrix[i, j] = 0 if self.matrix[i, j] == 1 else 1
        self.buttons[i][j].configure(bg="lightgray" if self.matrix[i, j] == 1 else "white")

    def save_matrix(self):
        print("Matrix saved:")
        print(self.matrix)
        # Here you can add code to save the matrix to a file or database as required.

def main():
    root = tk.Tk()
    app = MatrixDrawer(root)
    root.mainloop()

if __name__ == "__main__":
    main()
