import numpy as np
import pandas as pd
from scipy.stats import f
import matplotlib.pyplot as plt

def load_gene_data(file_path):
    data_matrix = pd.read_csv(file_path, sep=" ").values
    gene_data = []
    gene_symbols = []
    
    for record in data_matrix:
        gene_values = []
        record_parts = record[0].split('\t')
        gene_symbols.append(record_parts[49])
        record_parts.pop(0)
        
        for i in range(len(record_parts)):
            if i < 48:
                gene_values.append(float(record_parts[i]))
        
        gene_data.append(gene_values)
    
    return gene_data, gene_symbols

def generate_design_matrices():
    design_matrix_A = []
    design_matrix_A_hat = []
    
    for i in range(4):
        for j in range(12):
            if i == 0:
                design_matrix_A.append([0, 0, 1, 0])
                design_matrix_A_hat.append([0, 1, 1, 0])
            elif i == 1:
                design_matrix_A.append([1, 0, 0, 0])
                design_matrix_A_hat.append([1, 0, 1, 0])
            elif i == 2:
                design_matrix_A.append([0, 0, 0, 1])
                design_matrix_A_hat.append([0, 1, 0, 1])
            elif i == 3:
                design_matrix_A.append([0, 1, 0, 0])
                design_matrix_A_hat.append([1, 0, 0, 1])
    
    return design_matrix_A, design_matrix_A_hat

def calculate_p_values(gene_data, design_matrix_A, design_matrix_A_hat):
    p_values = []
    
    for record in gene_data:
        a = np.matmul(np.matmul(design_matrix_A, np.linalg.pinv(np.matmul(np.transpose(design_matrix_A), design_matrix_A))), np.transpose(design_matrix_A))
        b = np.matmul(np.matmul(design_matrix_A_hat, np.linalg.pinv(np.matmul(np.transpose(design_matrix_A_hat), design_matrix_A_hat))), np.transpose(design_matrix_A_hat))
        
        c = (np.matmul(np.matmul(np.transpose(record), np.subtract(a, b)), record)) / (np.matmul(np.matmul(np.transpose(record), np.subtract(np.identity(48), a)), record))
        dfn = 48 - np.linalg.matrix_rank(design_matrix_A)
        dfd = np.linalg.matrix_rank(design_matrix_A) - np.linalg.matrix_rank(design_matrix_A_hat)
        f_statistic = c * (dfn) / (dfd)
        p_value = f.cdf(f_statistic, dfd, dfn)
        p_values.append(1 - p_value)
    
    return p_values

def plot_histogram(p_values):
    plt.hist(p_values)
    print("Histogram saved as 'histogram.png' in the current directory.")
    plt.savefig('histogram.png')
    plt.show()

def filter_gene_symbols(p_values, gene_symbols):
    gene_shortlist = []
    
    for i in range(len(p_values)):
        gene_info = []
        
        if p_values[i] <= 0.05:
            gene_info.append(gene_symbols[i])
            gene_info.append(p_values[i])
            gene_shortlist.append(gene_info)
    
    return gene_shortlist

if __name__ == "__main__":
    input_file_path = "../data/Raw Data_GeneSpring.txt"
    gene_data, gene_symbols = load_gene_data(input_file_path)
  
    design_matrix_A, design_matrix_A_hat = generate_design_matrices()
    p_values = calculate_p_values(gene_data, design_matrix_A, design_matrix_A_hat)
    
    print("P values generated using a 2-way ANOVA framework:")
    print(p_values)
    print("\n")
    
    print("Draw the Histogram of p-values:")
    plot_histogram(p_values)
    print("\n")
    
    print("Shortlist rows:")
    gene_shortlist = filter_gene_symbols(p_values, gene_symbols)
    print(gene_shortlist)

