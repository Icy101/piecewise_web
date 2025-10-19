import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_and_plot_polynomial(file_path, degree, sheet_name=0):
    """
    Reads x and y from an Excel file, fits a polynomial of given degree,
    prints the equation, and plots the data and polynomial curve.
    """
    # --- Load Excel data ---
    data = pd.read_excel(file_path, sheet_name=sheet_name)
    if not {'x', 'y'}.issubset(data.columns):
        raise ValueError("Excel file must contain 'x' and 'y' columns.")
    
    x_data = data['x'].to_numpy()
    y_data = data['y'].to_numpy()
    
    # --- Fit polynomial ---
    coeffs = np.polyfit(x_data, y_data, degree)
    p = np.poly1d(coeffs)
    
    # --- Build readable equation string ---
    equation_terms = []
    for i, c in enumerate(coeffs):
        power = degree - i
        if abs(c) < 1e-10:
            continue
        term = f"{c:.4f}"
        if power > 0:
            term += f"*x^{power}" if power > 1 else "*x"
        equation_terms.append(term)
    equation = " + ".join(equation_terms)
    
    print("✅ Polynomial Equation Generated:")
    print(f"y = {equation}")
    
    # --- Generate smooth curve for plotting ---
    x_curve = np.linspace(x_data.min(), x_data.max(), 1000)
    y_curve = p(x_curve)
    
    # --- Plot the data and polynomial ---
    plt.figure(figsize=(10, 6))
    plt.scatter(x_data, y_data, color='red', label='Original Data', s=20)
    plt.plot(x_curve, y_curve, color='blue', linewidth=2, label=f'Polynomial Fit (degree={degree})')
    plt.title("Polynomial Fit to Excel Data")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend()
    plt.show()
    
    return coeffs, p

# --- Parameters ---
file_path = "Tensile Test Data.xlsx"  # Excel file name
degree = 300                    # Change this to whatever degree you want

# --- Run the generator ---
coeffs, poly_function = generate_and_plot_polynomial(file_path, degree)
