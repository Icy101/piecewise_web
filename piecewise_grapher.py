# ultimate_piecewise_grapher.py
import sys
import math
import numpy as np
import pandas as pd
from functools import partial

from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import curve_fit
try:
    # SciPy >=1.8 uses cumulative_trapezoid
    from scipy.integrate import cumulative_trapezoid as cumtrapz
except ImportError:
    from scipy.integrate import cumtrapz

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, Signal as pyqtSignal, QThread
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QMessageBox, QMainWindow, QWidget,
    QVBoxLayout, QPushButton, QLabel, QTabWidget, QProgressBar, QTextEdit, QHBoxLayout,
    QInputDialog   # <-- added
)

import matplotlib
matplotlib.use("Qt5Agg")  # For PySide6, you can keep Qt5Agg or use QtAgg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt  # <-- added to fix plt usage


import warnings
warnings.filterwarnings("ignore")

# ----------------- CONFIG -----------------
R2_THRESHOLD = 0.90        # threshold to accept a global fit
MAX_FIT_POINTS = 2000      # downsample points for model selection
MAX_PIECE_SEGMENTS = 10    # maximum piecewise segments shown
PLOT_POINTS = 2000         # points for smooth plotting

# ----------------- MODEL DEFINITIONS -----------------
def model_linear(x, a, b): return a*x + b
def model_quadratic(x, a, b, c): return a*x**2 + b*x + c
def model_cubic(x, a, b, c, d): return a*x**3 + b*x**2 + c*x + d
def model_quartic(x, a, b, c, d, e): return a*x**4 + b*x**3 + c*x**2 + d*x + e
def model_exponential(x, a, b, c): return a * np.exp(b * x) + c
def model_logarithmic(x, a, b, c): return a * np.log(np.abs(x) + 1e-12) + b * x + c
def model_sine(x, a, b, c, d): return a * np.sin(b * x + c) + d
def model_cosine(x, a, b, c, d): return a * np.cos(b * x + c) + d
def model_sigmoid(x, a, b, c, d): return a / (1 + np.exp(-b * (x - c))) + d

MODEL_SPECS = [
    ("Linear", model_linear, [1.0, 0.0]),
    ("Quadratic", model_quadratic, [1.0, 1.0, 0.0]),
    ("Cubic", model_cubic, [1.0, 1.0, 1.0, 0.0]),
    ("Quartic", model_quartic, [1.0, 1.0, 1.0, 1.0, 0.0]),
    ("Exponential", model_exponential, [1.0, 0.01, 0.0]),
    ("Logarithmic", model_logarithmic, [1.0, 0.0, 0.0]),
    ("Sine", model_sine, [1.0, 1.0, 0.0, 0.0]),
    ("Cosine", model_cosine, [1.0, 1.0, 0.0, 0.0]),
    ("Sigmoid", model_sigmoid, [1.0, 1.0, 0.5, 0.0]),
]

# ----------------- UTILITIES -----------------
def clean_and_aggregate(x, y):
    """Sort, drop NaN, and average duplicates"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]; y = y[mask]
    if x.size == 0:
        return x, y
    order = np.argsort(x)
    x = x[order]; y = y[order]
    ux, inv = np.unique(x, return_inverse=True)
    y_mean = np.array([y[inv == i].mean() for i in range(len(ux))])
    return ux, y_mean

def downsample(x, y, max_points=MAX_FIT_POINTS):
    n = len(x)
    if n <= max_points:
        return x, y
    idx = np.linspace(0, n-1, max_points, dtype=int)
    return x[idx], y[idx]

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1.0 - ss_res/ss_tot if ss_tot != 0 else -np.inf

def pretty_poly_from_coefs(coefs):
    parts = []
    deg = len(coefs) - 1
    for i, c in enumerate(coefs):
        if abs(c) < 1e-12:
            continue
        p = deg - i
        c_str = f"{c:.6g}"
        if p == 0:
            parts.append(f"{c_str}")
        elif p == 1:
            parts.append(f"{c_str}*x" if c_str != "1" else "x")
        else:
            parts.append(f"{c_str}*x^{p}" if c_str != "1" else f"x^{p}")
    return " + ".join(parts) if parts else "0"

def format_equation_text(model_name, popt, xs=None, ys=None):
    if model_name == "Piecewise":
        # compact piecewise using <= MAX_PIECE_SEGMENTS segments
        if xs is None or ys is None:
            return "Piecewise (no data)"
        n = len(xs)
        seg_count = min(MAX_PIECE_SEGMENTS, n-1)
        idx = np.linspace(0, n-1, seg_count+1, dtype=int)
        lines = []
        for i in range(len(idx)-1):
            a = idx[i]; b = idx[i+1]
            sx = xs[a:b+1]; sy = ys[a:b+1]
            if len(sx) < 3:
                slope = (sy[-1] - sy[0]) / (sx[-1] - sx[0] + 1e-12)
                intercept = sy[0] - slope * sx[0]
                expr = f"{slope:.6g}*x + {intercept:.6g}"
            else:
                coefs = np.polyfit(sx, sy, 2)  # quadratic on segment
                expr = pretty_poly_from_coefs(coefs)
            lines.append(f"Segment {i+1}: y = {expr}  for x in [{sx[0]:.6g}, {sx[-1]:.6g}]")
        return "\n".join(lines)
    else:
        if popt is None:
            return "No fit available"
        if model_name == "Linear":
            a,b = popt; return f"y = {a:.6g}*x + {b:.6g}"
        if model_name == "Quadratic":
            a,b,c = popt; return f"y = {a:.6g}*x^2 + {b:.6g}*x + {c:.6g}"
        if model_name == "Cubic":
            a,b,c,d = popt; return f"y = {a:.6g}*x^3 + {b:.6g}*x^2 + {c:.6g}*x + {d:.6g}"
        if model_name == "Quartic":
            a,b,c,d,e = popt; return f"y = {a:.6g}*x^4 + {b:.6g}*x^3 + {c:.6g}*x^2 + {d:.6g}*x + {e:.6g}"
        if model_name == "Exponential":
            a,b,c = popt; return f"y = {a:.6g} * e^({b:.6g}*x) + {c:.6g}"
        if model_name == "Logarithmic":
            a,b,c = popt; return f"y = {a:.6g}*ln(|x|) + {b:.6g}*x + {c:.6g}"
        if model_name == "Sine":
            a,b,c,d = popt; return f"y = {a:.6g}*sin({b:.6g}*x + {c:.6g}) + {d:.6g}"
        if model_name == "Cosine":
            a,b,c,d = popt; return f"y = {a:.6g}*cos({b:.6g}*x + {c:.6g}) + {d:.6g}"
        if model_name == "Sigmoid":
            a,b,c,d = popt; return f"y = {a:.6g}/(1 + e^(-{b:.6g}*(x - {c:.6g}))) + {d:.6g}"
        return "Unknown model"

# ----------------- MODEL SELECTION & FITTING -----------------
def safe_curve_fit(func, x, y, p0=None, maxfev=50000):
    try:
        popt, _ = curve_fit(func, x, y, p0=p0, maxfev=maxfev)
        return popt
    except Exception:
        return None

def heuristic_preferred_models(x, y):
    dx = np.diff(x)
    dy = np.diff(y)
    slopes = np.abs(dy / (dx + 1e-12))
    slope_var = np.nanstd(slopes) / (np.nanmean(slopes) + 1e-12)
    if slope_var < 0.12:
        return ["Linear","Quadratic","Cubic","Quartic"]
    if np.nanmin(y) > 0 and (np.nanmax(y) / (np.nanmin(y) + 1e-12)) > 10:
        return ["Exponential","Sigmoid"]
    if (np.nanmax(y) - np.nanmin(y)) / (np.nanmean(np.abs(y)) + 1e-12) > 0.5 and slope_var > 0.6:
        return ["Sine","Cosine"]
    return [name for name,_,_ in MODEL_SPECS]

def choose_best_global_model(x, y, r2_threshold=R2_THRESHOLD):
    xs, ys = downsample(x, y, max_points=MAX_FIT_POINTS)
    preferred = heuristic_preferred_models(xs, ys)
    best_name, best_popt, best_r2 = None, None, -np.inf
    for name, func, p0 in MODEL_SPECS:
        if name not in preferred:
            continue
        popt = safe_curve_fit(func, xs, ys, p0=p0)
        if popt is None:
            continue
        try:
            yp = func(xs, *popt)
            r2 = r_squared(ys, yp)
        except Exception:
            continue
        if r2 > best_r2:
            best_name, best_popt, best_r2 = name, popt, r2
    # fallback: try all if none in preferred succeeded
    if best_name is None:
        for name, func, p0 in MODEL_SPECS:
            popt = safe_curve_fit(func, xs, ys, p0=p0)
            if popt is None: continue
            try:
                yp = func(xs, *popt); r2 = r_squared(ys, yp)
            except Exception:
                continue
            if r2 > best_r2:
                best_name, best_popt, best_r2 = name, popt, r2
    # decide acceptance
    if best_r2 >= r2_threshold:
        return best_name, best_popt, best_r2
    # else fallback to piecewise
    return "Piecewise", None, best_r2

# ----------------- THREAD WORKER -----------------
class FitWorker(QThread):
    finished_signal = pyqtSignal(dict)  # sends results dict
    progress_signal = pyqtSignal(int)

    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y

    def run(self):
        results = {}
        try:
            # Clean data
            x_clean, y_clean = clean_and_aggregate(self.x, self.y)
            results['n_points'] = len(x_clean)
            self.progress_signal.emit(10)

            # Spline for original curve
            spline = InterpolatedUnivariateSpline(x_clean, y_clean, k=3)
            x_smooth = np.linspace(x_clean[0], x_clean[-1], PLOT_POINTS)
            y_smooth = spline(x_smooth)
            results['spline_x'] = x_smooth; results['spline_y'] = y_smooth
            self.progress_signal.emit(25)

            # Best global model for spline curve
            model_name, popt, r2 = choose_best_global_model(x_smooth, y_smooth)
            results['spline_model'] = (model_name, popt, r2)
            self.progress_signal.emit(40)

            # Derivative (numerical)
            dy = np.gradient(y_clean, x_clean)
            results['der_x'] = x_clean; results['der_y'] = dy
            der_model, der_popt, der_r2 = choose_best_global_model(x_clean, dy)
            results['der_model'] = (der_model, der_popt, der_r2)
            self.progress_signal.emit(60)

            # Integral (numerical)
            yi = cumtrapz(y_clean, x_clean, initial=0.0)
            results['int_x'] = x_clean; results['int_y'] = yi
            int_model, int_popt, int_r2 = choose_best_global_model(x_clean, yi)
            results['int_model'] = (int_model, int_popt, int_r2)
            self.progress_signal.emit(85)

            # Pack final info
            results['x_clean'] = x_clean; results['y_clean'] = y_clean
            self.progress_signal.emit(100)
        except Exception as e:
            results['error'] = str(e)
        self.finished_signal.emit(results)

# ----------------- MATPLOTLIB WIDGET -----------------
class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        fig = Figure(figsize=(6, 4), dpi=100)
        self.ax = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)

# ----------------- MAIN WINDOW -----------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ultimate Piecewise Grapher")
        self.resize(1200, 800)

        # Main tab widget
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Placeholder tabs for Spline, Derivative, Integral
        self.spline_tab = QWidget()
        self.derivative_tab = QWidget()
        self.integral_tab = QWidget()

        self.tabs.addTab(self.spline_tab, "Spline")
        self.tabs.addTab(self.derivative_tab, "Derivative")
        self.tabs.addTab(self.integral_tab, "Integral")

        # Layouts for tabs
        self.spline_layout = QVBoxLayout(self.spline_tab)
        self.derivative_layout = QVBoxLayout(self.derivative_tab)
        self.integral_layout = QVBoxLayout(self.integral_tab)

        # Buttons to load Excel data
        # Buttons to load Excel data (one per tab)
        self.load_button_spline = QPushButton("Load Excel File")
        self.load_button_derivative = QPushButton("Load Excel File")
        self.load_button_integral = QPushButton("Load Excel File")

        for btn in [self.load_button_spline, self.load_button_derivative, self.load_button_integral]:
            btn.setFixedHeight(40)
            btn.setStyleSheet("background-color: #3498db; color: white; font-size: 16px;")
            btn.clicked.connect(self.load_excel)

        self.spline_layout.addWidget(self.load_button_spline)
        self.derivative_layout.addWidget(self.load_button_derivative)
        self.integral_layout.addWidget(self.load_button_integral)

        # Buttons to show piecewise equations
        # Buttons for piecewise / best-fit (hidden initially)
        self.piecewise_button_spline = QPushButton("Show Piecewise Equation")
        self.bestfit_button_spline = QPushButton("Show Best-Fit Equation")
        self.piecewise_button_derivative = QPushButton("Show Piecewise Equation")
        self.bestfit_button_derivative = QPushButton("Show Best-Fit Equation")
        self.piecewise_button_integral = QPushButton("Show Piecewise Equation")
        self.bestfit_button_integral = QPushButton("Show Best-Fit Equation")

        all_buttons = [
            self.piecewise_button_spline, self.bestfit_button_spline,
            self.piecewise_button_derivative, self.bestfit_button_derivative,
            self.piecewise_button_integral, self.bestfit_button_integral
        ]

        for btn in all_buttons:
            btn.setFixedHeight(35)
            btn.setStyleSheet("background-color: #2ecc71; color: white; font-size: 14px;")
            btn.hide()  # hidden initially

        # Add to layouts
        self.spline_layout.addWidget(self.piecewise_button_spline)
        self.spline_layout.addWidget(self.bestfit_button_spline)
        self.derivative_layout.addWidget(self.piecewise_button_derivative)
        self.derivative_layout.addWidget(self.bestfit_button_derivative)
        self.integral_layout.addWidget(self.piecewise_button_integral)
        self.integral_layout.addWidget(self.bestfit_button_integral)

        # Connect signals
        self.piecewise_button_spline.clicked.connect(lambda: self.show_piecewise("Spline"))
        self.bestfit_button_spline.clicked.connect(lambda: self.show_bestfit("Spline"))
        self.piecewise_button_derivative.clicked.connect(lambda: self.show_piecewise("Derivative"))
        self.bestfit_button_derivative.clicked.connect(lambda: self.show_bestfit("Derivative"))
        self.piecewise_button_integral.clicked.connect(lambda: self.show_piecewise("Integral"))
        self.bestfit_button_integral.clicked.connect(lambda: self.show_bestfit("Integral"))


        # Text widgets to display equations
        self.spline_text = QTextEdit()
        self.spline_text.setReadOnly(True)
        self.derivative_text = QTextEdit()
        self.derivative_text.setReadOnly(True)
        self.integral_text = QTextEdit()
        self.integral_text.setReadOnly(True)

        self.spline_layout.addWidget(self.spline_text)
        self.derivative_layout.addWidget(self.derivative_text)
        self.integral_layout.addWidget(self.integral_text)

        # Figure placeholders
        self.spline_canvas = None
        self.derivative_canvas = None
        self.integral_canvas = None

        # Data
        self.x_data = None
        self.y_data = None
        self.max_segments = 10

    def show_piecewise(self, tab_name):
        if tab_name == "Spline":
            x = np.linspace(min(self.x_data), max(self.x_data), 1000)
            y = InterpolatedUnivariateSpline(self.x_data, self.y_data, k=3)(x)
            eq_text = self.piecewise_equations(x, y, self.max_segments, "y")
            self.spline_text.setPlainText(eq_text)

        elif tab_name == "Derivative":
            dy = np.diff(self.y_data)
            dx = np.diff(self.x_data)
            x = (self.x_data[:-1] + self.x_data[1:]) / 2
            y = dy / dx
            eq_text = self.piecewise_equations(x, y, self.max_segments, "dy/dx")
            self.derivative_text.setPlainText(eq_text)

        elif tab_name == "Integral":
            x = self.x_data
            y = np.concatenate([[0], np.cumsum((self.y_data[:-1] + self.y_data[1:]) / 2 * np.diff(self.x_data))])
            eq_text = self.piecewise_equations(x, y, self.max_segments, "∫y dx")
            self.integral_text.setPlainText(eq_text)


    def load_excel(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Excel File", "", "Excel Files (*.xlsx *.xls)")
        if not file_path:
            return

        try:
            df = pd.read_excel(file_path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open file:\n{e}")
            return

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) < 2:
            QMessageBox.warning(self, "Warning", "File must have at least two numeric columns.")
            return

        # Ask user to select columns
        x_col, ok1 = QInputDialog.getItem(self, "Select X Column", "Column for X:", numeric_cols, 0, False)
        if not ok1:
            return
        y_col, ok2 = QInputDialog.getItem(self, "Select Y Column", "Column for Y:", numeric_cols, 1, False)
        if not ok2:
            return

        # Load and clean data
        x = df[x_col].values
        y = df[y_col].values
        sort_idx = np.argsort(x)
        x = x[sort_idx]
        y = y[sort_idx]

        # Remove duplicates
        unique_x, indices = np.unique(x, return_inverse=True)
        y_unique = np.array([y[indices == i].mean() for i in range(len(unique_x))])
        self.x_data, self.y_data = unique_x, y_unique

        # Update all tabs
        self.update_spline_tab()
        self.update_derivative_tab()
        self.update_integral_tab()
        # Show piecewise / best-fit buttons after loading data
        for btn in [
            self.piecewise_button_spline, self.bestfit_button_spline,
            self.piecewise_button_derivative, self.bestfit_button_derivative,
            self.piecewise_button_integral, self.bestfit_button_integral
        ]:
            btn.show()

    def show_bestfit(self, tab_name):
        if tab_name == "Spline":
            x_smooth = np.linspace(min(self.x_data), max(self.x_data), 1000)
            spline = InterpolatedUnivariateSpline(self.x_data, self.y_data, k=3)
            y_smooth = spline(x_smooth)
            func_name, coefs, r2 = self.fit_best_function(x_smooth, y_smooth)
            eq_text = f"Best Fit: {func_name} (R²={r2:.3f})\nEquation: {self.format_equation(func_name, coefs, 'y')}"
            self.spline_text.setPlainText(eq_text)

        elif tab_name == "Derivative":
            dy = np.diff(self.y_data)
            dx = np.diff(self.x_data)
            x = (self.x_data[:-1] + self.x_data[1:]) / 2
            y = dy / dx
            func_name, coefs, r2 = self.fit_best_function(x, y)
            eq_text = f"Best Fit: {func_name} (R²={r2:.3f})\nEquation: {self.format_equation(func_name, coefs, 'dy/dx')}"
            self.derivative_text.setPlainText(eq_text)

        elif tab_name == "Integral":
            x = self.x_data
            y = np.concatenate([[0], np.cumsum((self.y_data[:-1] + self.y_data[1:]) / 2 * np.diff(self.x_data))])
            func_name, coefs, r2 = self.fit_best_function(x, y)
            eq_text = f"Best Fit: {func_name} (R²={r2:.3f})\nEquation: {self.format_equation(func_name, coefs, '∫y dx')}"
            self.integral_text.setPlainText(eq_text)


    def update_spline_tab(self):
        self.spline_text.clear()
        if self.spline_canvas:
            self.spline_layout.removeWidget(self.spline_canvas)
            self.spline_canvas.setParent(None)

    # Create spline
        spline = InterpolatedUnivariateSpline(self.x_data, self.y_data, k=3)

    # Fit best global model to spline points
        x_smooth = np.linspace(min(self.x_data), max(self.x_data), 1000)
        y_smooth = spline(x_smooth)
        func_name, coefs, r2 = self.fit_best_function(x_smooth, y_smooth)
        if r2 > 0.8:
            eq = self.format_equation(func_name, coefs, "y")
            self.spline_text.append(f"Best Fit: {func_name} (R²={r2:.3f})\nEquation: {eq}")
        else:
            eqs = self.piecewise_equations(x_smooth, y_smooth, self.max_segments, "y")
            self.spline_text.append(eqs)

        self.plot_function(spline, self.x_data, self.y_data, self.spline_layout, self.spline_text, "Spline")


    def update_derivative_tab(self):
        self.derivative_text.clear()
        if self.derivative_canvas:
            self.derivative_layout.removeWidget(self.derivative_canvas)
            self.derivative_canvas.setParent(None)

        # Numerical derivative
        dy = np.diff(self.y_data)
        dx = np.diff(self.x_data)
        derivative_values = dy / dx
        derivative_x = (self.x_data[:-1] + self.x_data[1:]) / 2

        # Fit simple function
        func_name, coeffs, r2 = self.fit_best_function(derivative_x, derivative_values)
        if r2 > 0.8:
            eq = self.format_equation(func_name, coeffs, "dy/dx")
            self.derivative_text.append(f"Best Fit: {func_name} (R²={r2:.3f})\nEquation: {eq}")
        else:
            # Piecewise
            eqs = self.piecewise_equations(derivative_x, derivative_values, self.max_segments, "dy/dx")
            self.derivative_text.append(eqs)

        spline_deriv = InterpolatedUnivariateSpline(derivative_x, derivative_values, k=3)
        self.plot_function(spline_deriv, derivative_x, derivative_values, self.derivative_layout, self.derivative_text, "Derivative")

    def update_integral_tab(self):
        self.integral_text.clear()
        if self.integral_canvas:
            self.integral_layout.removeWidget(self.integral_canvas)
            self.integral_canvas.setParent(None)

        # Trapezoidal integral
        integral_values = np.concatenate([[0], np.cumsum((self.y_data[:-1] + self.y_data[1:]) / 2 * np.diff(self.x_data))])
        integral_x = self.x_data

        func_name, coeffs, r2 = self.fit_best_function(integral_x, integral_values)
        if r2 > 0.8:
            eq = self.format_equation(func_name, coeffs, "∫y dx")
            self.integral_text.append(f"Best Fit: {func_name} (R²={r2:.3f})\nEquation: {eq}")
        else:
            eqs = self.piecewise_equations(integral_x, integral_values, self.max_segments, "∫y dx")
            self.integral_text.append(eqs)

        spline_int = InterpolatedUnivariateSpline(integral_x, integral_values, k=3)
        self.plot_function(spline_int, integral_x, integral_values, self.integral_layout, self.integral_text, "Integral")

    def plot_function(self, func, x_data, y_data, layout, text_widget, title):
        fig, ax = plt.subplots(figsize=(8,5))
        ax.plot(x_data, y_data, 'bo', label="Data")
        x_vals = np.linspace(min(x_data), max(x_data), 1000)
        y_vals = func(x_vals)
        ax.plot(x_vals, y_vals, 'r-', label=title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True)
        ax.legend()
        canvas = FigureCanvasQTAgg(fig)
        layout.addWidget(canvas)
        if title == "Spline":
            self.spline_canvas = canvas
        elif title == "Derivative":
            self.derivative_canvas = canvas
        elif title == "Integral":
            self.integral_canvas = canvas
        canvas.draw()

    def fit_best_function(self, x, y):
        # Try polynomial, exponential, log, sine
        best_r2 = -np.inf
        best_name = "Piecewise"
        best_coefs = None

        # Polynomial fit
        for deg in [1,2,3,4]:
            try:
                coefs = np.polyfit(x, y, deg)
                y_pred = np.polyval(coefs, x)
                r2 = 1 - np.sum((y - y_pred)**2)/np.sum((y - np.mean(y))**2)
                if r2 > best_r2:
                    best_r2 = r2
                    best_name = f"Polynomial deg={deg}"
                    best_coefs = coefs
            except:
                continue
        return best_name, best_coefs, best_r2

    def piecewise_equations(self, x, y, max_segments, symbol="y"):
        n = len(x)
        seg_len = max(1, n // max_segments)
        eqs = f"Piecewise approximation (max {max_segments} segments):\n"
        for i in range(0, n-1, seg_len):
            xi0 = x[i]
            xi1 = x[min(i+seg_len, n-1)]
            xs = x[i:min(i+seg_len+1, n)]
            ys = y[i:min(i+seg_len+1, n)]
            coefs = np.polyfit(xs, ys, 1 if len(xs)<3 else 2)
            eq = self.format_polynomial(coefs, symbol)
            eqs += f"Segment {i//seg_len+1}: {eq} for x in [{xi0:.3g}, {xi1:.3g}]\n"
        return eqs

    def format_polynomial(self, coefs, symbol="y"):
        terms = []
        deg = len(coefs)-1
        for i, c in enumerate(coefs):
            if abs(c)<1e-10:
                continue
            power = deg-i
            if power == 0:
                terms.append(f"{c:.6g}")
            elif power == 1:
                terms.append(f"{c:.6g}*x")
            else:
                terms.append(f"{c:.6g}*x^{power}")
        return f"{symbol} = " + " + ".join(terms)

    def format_equation(self, func_name, coefs, symbol="y"):
        if "Polynomial" in func_name:
            return self.format_polynomial(coefs, symbol)
        return f"{symbol} = piecewise (complex function)"
    

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

