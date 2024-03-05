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




# modify this code so that each time I can draw two 12 x 12 matrices simultaneously, and save them to a file.