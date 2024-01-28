import statistics
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import Scale, Button, ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from reportlab.pdfgen import canvas
from io import BytesIO
from matplotlib.colors import Normalize


class TumorSimulatorApp:
    def __init__(self, master):
        self.master = master
        master.title("Tumor Simulator")

        # Notebook for organizing sections
        self.notebook = ttk.Notebook(master)
        self.notebook.pack(fill=tk.BOTH, expand=1)
        # Additional variable for storing the graphs plot
        self.graphs_plot = None

        # First tab for tumor simulation
        self.tumor_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.tumor_tab, text='Tumor Simulation')

        # Second tab for simulation controls
        self.controls_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.controls_tab, text='Simulation Controls')

        
        self.tumor_size = 20
        self.speed = 100  # Initial speed (milliseconds)
        self.generations = int(60000 / self.speed)
        self.tumor_growth_probability = 0.01  # Initial tumor growth probability
        self.tumor_death_probability = 0.34  # Initial tumor cell death probability
        self.living_death_probability = 0.009  # Initial living cell death probability
        self.regeneration_probability = 0.1  # Initial regeneration probability
        self.initial_tumor = self.initialize_tumor(self.tumor_size)
        # Analytical solution variables
        self.analytical_solution_time_points = []
        self.analytical_solution_tumor_sizes = []
        # Simulation solution variables
        self.simulation_time_points = []
        self.simulation_tumor_sizes = []       
        
        # Tumor simulation tab
        self.fig, self.ax = plt.subplots()
        self.rectangles = []
        self.monte_carlo_results = []

        # ...

        self.monte_carlo_button = Button(self.controls_tab, text="Run Monte Carlo Simulation (5 runs)",
                                         command=self.run_monte_carlo_simulation)
        self.monte_carlo_button.pack()

        for i in range(self.tumor_size):
            for j in range(self.tumor_size):
                rect = patches.Rectangle((i - 0.5, j - 0.5), 1, 1, edgecolor='pink', linewidth=1, facecolor='none')
                self.rectangles.append(rect)
                self.ax.add_patch(rect)

        self.img = self.ax.imshow(self.initial_tumor, cmap='Reds', interpolation='none', vmin=0, vmax=2)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.tumor_tab)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Simulation controls tab
        self.speed_scale = Scale(self.controls_tab, from_=50, to=1000, orient=tk.HORIZONTAL, label="Speed", length=300,
                                 resolution=50, command=self.update_speed)
        self.speed_scale.set(self.speed)
        self.speed_scale.pack()

        self.probability_scale = Scale(self.controls_tab, from_=0, to=1, orient=tk.HORIZONTAL,
                                       label="Tumor Growth Probability", length=300, resolution=0.01,
                                       command=self.update_probability)
        self.probability_scale.set(self.tumor_growth_probability)
        self.probability_scale.pack()

        self.death_probability_scale = Scale(self.controls_tab, from_=0, to=1, orient=tk.HORIZONTAL,
                                             label="Tumor Cell Death Probability", length=300, resolution=0.01,
                                             command=self.update_death_probability)
        self.death_probability_scale.set(self.tumor_death_probability)
        self.death_probability_scale.pack()

        self.living_death_probability_scale = Scale(self.controls_tab, from_=0, to=1, orient=tk.HORIZONTAL,
                                                    label="Living Cell Death Probability", length=300, resolution=0.01,
                                                    command=self.update_living_death_probability)
        self.living_death_probability_scale.set(self.living_death_probability)
        self.living_death_probability_scale.pack()

        self.regeneration_probability_scale = Scale(self.controls_tab, from_=0, to=1, orient=tk.HORIZONTAL,
                                                    label="Regeneration Probability", length=300, resolution=0.01,
                                                    command=self.update_regeneration_probability)
        self.regeneration_probability_scale.set(self.regeneration_probability)
        self.regeneration_probability_scale.pack()

        self.start_button = Button(self.controls_tab, text="Start Simulation", command=self.start_simulation)
        self.start_button.pack()

        self.show_graphs_button = Button(self.controls_tab, text="Show Graphs", command=self.show_graphs)
        self.show_graphs_button.pack()

        self.reset_button = Button(self.controls_tab, text="Reset Simulation", command=self.reset_simulation)
        self.reset_button.pack()

  

        self.animation = None
        self.tumor_counts = {'tumor_cells': [], 'dead_tumor_cells': [], 'living_cells': []}

    def initialize_tumor(self, size):
        tumor = np.zeros((size, size))
        random_row, random_col = np.random.randint(0, size), np.random.randint(0, size)
        tumor[random_row, random_col] = 2
        tumor[tumor == 4] = 0
        return tumor

    def update_tumor(self, tumor, frame):
        new_tumor = np.copy(tumor)
        rows, cols = tumor.shape
        test = np.copy(tumor)

        for i in range(rows):
            for j in range(cols):

                if new_tumor[i, j] == 2:
                    if new_tumor[i, j] == 2 and np.random.rand() < self.tumor_death_probability:
                        test[i, j] = 3

                    # if np.random.rand() < self.tumor_growth_probability:
                        # direction = np.random.choice(['N', 'S', 'W', 'E', 'NE', 'NW', 'SE', 'SW'])

                        # if 'N' in direction and i - 1 >= 0:
                        #     if new_tumor[i - 1, j] == 0:
                        #         test[i - 1, j] = 2
                        # if 'S' in direction and i + 1 < rows:
                        #     if new_tumor[i + 1, j] == 0:
                        #         test[i + 1, j] = 2
                        # if 'W' in direction and j - 1 >= 0:
                        #     if new_tumor[i, j - 1] == 0:
                        #         test[i, j - 1] = 2
                        # if 'E' in direction and j + 1 < cols:
                        #     if new_tumor[i, j + 1] == 0:
                        #         test[i, j + 1] = 2
                        # if 'NE' in direction and i - 1 >= 0 and j + 1 < cols:
                        #     if new_tumor[i - 1, j + 1] == 0:
                        #         test[i - 1, j + 1] = 2
                        # if 'NW' in direction and i - 1 >= 0 and j - 1 >= 0:
                        #     if new_tumor[i - 1, j - 1] == 0:
                        #         test[i - 1, j - 1] = 2
                        # if 'SE' in direction and i + 1 < rows and j + 1 < cols:
                        #     if new_tumor[i + 1, j + 1] == 0:
                        #         test[i + 1, j + 1] = 2
                        # if 'SW' in direction and i + 1 < rows and j - 1 >= 0:
                        #     if new_tumor[i + 1, j - 1] == 0:
                        #         test[i + 1, j - 1] = 2

                elif new_tumor[i, j] == 0:
                
                    neighbour=0
                    if  i - 1 >= 0:
                        if new_tumor[i - 1, j] == 2:
                            neighbour=neighbour+1
                    if  i + 1 < rows:
                        if new_tumor[i + 1, j] == 2:
                            neighbour=neighbour+1
                    if  j - 1 >= 0:
                        if new_tumor[i, j - 1] == 2:
                            neighbour=neighbour+1
                    if  j + 1 < cols:
                        if new_tumor[i, j + 1] == 2:
                            neighbour=neighbour+1
                    if  i - 1 >= 0 and j + 1 < cols:
                        if new_tumor[i - 1, j + 1] == 2:
                            neighbour=neighbour+1
                    if  i - 1 >= 0 and j - 1 >= 0:
                        if new_tumor[i - 1, j - 1] == 2:
                                neighbour=neighbour+1
                    if  i + 1 < rows and j + 1 < cols:
                        if new_tumor[i + 1, j + 1] == 2:
                            neighbour=neighbour+1
                    if  i + 1 < rows and j - 1 >= 0:
                        if new_tumor[i + 1, j - 1] == 2:
                            neighbour=neighbour+1
                    if  neighbour != 0:

                        neighbour_effect = self.tumor_growth_probability + (neighbour/8)
                        mapped_specific_probability_tumor = (neighbour_effect - self.tumor_growth_probability) * (1-self.tumor_growth_probability) + self.tumor_growth_probability
                        if np.random.rand() < mapped_specific_probability_tumor:
                            test[i, j] = 2
                    
                    elif np.random.rand() < self.living_death_probability:
                        test[i, j] = 4

            
                elif new_tumor[i, j] == 4 or new_tumor[i, j] == 3:
                    if np.random.rand() < self.regeneration_probability:
                        test[i, j] = 0

        return test

    def animate_tumor(self, frame):
        new_tumor = self.update_tumor(self.initial_tumor, frame)
        changed_indices = np.where(new_tumor != self.initial_tumor)
        for i, j in zip(changed_indices[0], changed_indices[1]):
            if new_tumor[i, j] == 3:
                self.rectangles[i * self.tumor_size + j].set_edgecolor('black')
                self.rectangles[i * self.tumor_size + j].set_facecolor('black')
            if new_tumor[i, j] == 4:
                self.rectangles[i * self.tumor_size + j].set_edgecolor('black')
                self.rectangles[i * self.tumor_size + j].set_facecolor('blue')
            if new_tumor[i, j] == 0:
                self.rectangles[i * self.tumor_size + j].set_edgecolor('black')
                self.rectangles[i * self.tumor_size + j].set_facecolor('white')
            if new_tumor[i, j] == 2:
                self.rectangles[i * self.tumor_size + j].set_edgecolor('black')
                self.rectangles[i * self.tumor_size + j].set_facecolor('red')

            self.rectangles[i * self.tumor_size + j].set_xy((j - 0.5, i - 0.5))

        tumor_count = np.count_nonzero(new_tumor == 2)
        dead_tumor_count = np.count_nonzero(new_tumor == 3)
        living_cells_count = np.count_nonzero(new_tumor == 0)

        self.tumor_counts['tumor_cells'].append(tumor_count)
        self.tumor_counts['dead_tumor_cells'].append(dead_tumor_count)
        self.tumor_counts['living_cells'].append(living_cells_count)

        self.initial_tumor = new_tumor
        self.img.set_array(self.initial_tumor)

        return self.img, *self.rectangles

    def end_simulation(self):
        fig, ax = plt.subplots()
        frames = np.arange(1, min(self.generations + 1, len(self.tumor_counts['tumor_cells']) + 1))

        ax.plot(frames, self.tumor_counts['tumor_cells'][:len(frames)], label='Tumor Cells')
        ax.plot(frames, self.tumor_counts['dead_tumor_cells'][:len(frames)], label='Dead Tumor Cells')
        ax.plot(frames, self.tumor_counts['living_cells'][:len(frames)], label='Living Cells')

        ax.set_xlabel('Generation')
        ax.set_ylabel('Cell Count')
        ax.legend()

        plt.show()

    def start_simulation(self):
        if self.animation is not None:
            self.animation.event_source.stop()
        self.animation = animation.FuncAnimation(self.fig, self.animate_tumor, frames=self.generations,
                                                interval=self.speed, blit=True)
        


    def reset_simulation(self):
        if self.animation is not None:
            self.animation.event_source.stop()
        self.initial_tumor = self.initialize_tumor(self.tumor_size)
        self.img.set_array(self.initial_tumor)

        for i, rect in enumerate(self.rectangles):
            row, col = divmod(i, self.tumor_size)
            rect.set_xy((col - 0.5, row - 0.5))
            rect.set_edgecolor('pink')
            rect.set_facecolor('none')

        self.tumor_counts = {'tumor_cells': [], 'dead_tumor_cells': [], 'living_cells': []}
        self.canvas.draw()

    def show_graphs(self):
        if self.graphs_plot is not None:
            self.graphs_plot.destroy()  # Close the existing plot if any

        fig, ax = plt.subplots()
        frames = np.arange(1, len(self.tumor_counts['tumor_cells']) + 1)

        ax.plot(frames, self.tumor_counts['tumor_cells'], label='Tumor Cells')
        ax.plot(frames, self.tumor_counts['dead_tumor_cells'], label='Dead Tumor Cells')
        ax.plot(frames, self.tumor_counts['living_cells'], label='Living Cells')

        ax.set_xlabel('Generation')
        ax.set_ylabel('Cell Count')
        ax.legend()

        # Create a new window for the graphs plot
        self.graphs_plot = tk.Toplevel(self.master)
        self.graphs_plot.wm_title("Simulation Graphs")
        self.graphs_plot.protocol("WM_DELETE_WINDOW", self.close_graphs_plot)

        canvas = FigureCanvasTkAgg(fig, master=self.graphs_plot)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def close_graphs_plot(self):
        if self.graphs_plot is not None:
            self.graphs_plot.destroy()
            self.graphs_plot = None

    def update_speed(self, val):
        self.speed = int(val)
        if self.animation is not None:
            self.animation.event_source.interval = self.speed

    def update_probability(self, val):
        self.tumor_growth_probability = float(val)

    def update_death_probability(self, val):
        self.tumor_death_probability = float(val)

    def update_living_death_probability(self, val):
        self.living_death_probability = float(val)

    def update_regeneration_probability(self, val):
        self.regeneration_probability = float(val)


    def compute_analytical_solution(self):
        # Reset previous analytical solution data
        self.analytical_solution_time_points = []
        self.analytical_solution_tumor_sizes = []

        # Initial conditions
        tn = 0
        Pn = 1  # Initial tumor size

        # Time step for Euler's method
        delta_t = 1

        # Number of time steps (adjust as needed)
        num_steps = 100

        for _ in range(num_steps):
            # Update time and tumor size using Euler's method
            tn = tn + delta_t
            Pn = Pn + self.tumor_growth_probability * Pn * delta_t - self.tumor_death_probability * Pn * delta_t

            # Store results for plotting
            self.analytical_solution_time_points.append(tn)
            self.analytical_solution_tumor_sizes.append(Pn)
           

        # Plot the analytical solution
        plt.figure()
        plt.plot(self.analytical_solution_time_points, self.analytical_solution_tumor_sizes, label='Analytical Solution')
        plt.xlabel('Time')
        plt.ylabel('Tumor Size')
        plt.title('Analytical Solution using Euler\'s Method')
        plt.legend()
        plt.show()

    def run_monte_carlo_simulation(self):
        pdf_buffer = BytesIO()
        pdf_pages = PdfPages(pdf_buffer)

        results = []

        for run_index in range(10):
            tumor_counts = {'tumor_cells': [], 'dead_tumor_cells': [], 'living_cells': []}
            initial_tumor = self.initialize_tumor(self.tumor_size)

            for frame in range(self.generations):
                new_tumor = self.update_tumor(initial_tumor, frame)

                tumor_count = np.count_nonzero(new_tumor == 2)
                dead_tumor_count = np.count_nonzero(new_tumor == 3)
                living_cells_count = np.count_nonzero(new_tumor == 0)

                tumor_counts['tumor_cells'].append(tumor_count)
                tumor_counts['dead_tumor_cells'].append(dead_tumor_count)
                tumor_counts['living_cells'].append(living_cells_count)

                initial_tumor = new_tumor

            results.append(tumor_counts)

            # Generate visualizations for each run
            self.generate_visualizations(run_index, initial_tumor, tumor_counts, pdf_pages)

        # Generate summary page
        self.generate_summary_page(results, pdf_pages)

        # Close the PdfPages instance before saving the buffer to a file
        pdf_pages.close()

        # Save the PDF buffer to a file or display as needed
        with open('monte_carlo_results.pdf', 'wb') as pdf_file:
            pdf_file.write(pdf_buffer.getvalue())
        pdf_buffer.close()

    def generate_visualizations(self, run_index, final_grid, tumor_counts, pdf_pages):
        # Generate visualizations and add to the PDF
        
        plt.figure()

        # Create a custom colormap for visualization
        custom_cmap = ListedColormap(['white','white','red','black','blue'])

        plt.imshow(final_grid, cmap=custom_cmap, interpolation='none')
        plt.title(f'Final Grid - Run {run_index + 1}')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        pdf_pages.savefig()
        plt.close()

        plt.figure()
        plt.plot(tumor_counts['tumor_cells'], label='Tumor Cells')
        plt.plot(tumor_counts['dead_tumor_cells'], label='Dead Tumor Cells')
        plt.plot(tumor_counts['living_cells'], label='Living Cells')
        plt.xlabel('Generation')
        plt.ylabel('Cell Count')
        plt.title(f'Run {run_index + 1} - Cell Counts Over Generations')
        plt.legend()
        pdf_pages.savefig()
        plt.close()
    def generate_summary_page(self, results, pdf_pages):
        # Extract the tumor cell counts at the end of each run
        end_tumor_counts = [result['tumor_cells'][-1] for result in results]

        # Calculate mean and standard deviation
        mean_end_tumor_count = statistics.mean(end_tumor_counts)
        std_dev_end_tumor_count = statistics.stdev(end_tumor_counts)

        # Create a summary figure
        plt.figure()
        plt.bar(range(len(end_tumor_counts)), end_tumor_counts, color='lightblue', label='Tumor Cells at End of Run')
        plt.axhline(mean_end_tumor_count, color='red', linestyle='dashed', linewidth=2, label='Mean')
        plt.xlabel('Run Index')
        plt.ylabel('Cell Count')
        plt.title('Summary - Tumor Cells at End of Each Run')
        plt.legend()

        # Save the summary figure to the PDF
        pdf_pages.savefig()
        plt.close()

        # Display mean and standard deviation
        print(f"Mean of Tumor Cells at End: {mean_end_tumor_count}")
        print(f"Standard Deviation of Tumor Cells at End: {std_dev_end_tumor_count}")

        # Add numerical results to the next page
        self.add_numerical_results_page(mean_end_tumor_count, std_dev_end_tumor_count, pdf_pages)

    def add_numerical_results_page(self, mean_value, std_dev_value, pdf_pages):
        # Create a new page for numerical results
        plt.figure()
        plt.text(0.5, 0.5, f"Mean of Tumor Cells at End: {mean_value}\nStandard Deviation of Tumor Cells at End: {std_dev_value}",
                ha='center', va='center', fontsize=12)
        plt.axis('off')
        plt.title('Numerical Results')
        
        # Save the numerical results page to the PDF
        pdf_pages.savefig()
        plt.close()
    

if __name__ == "__main__":
    root = tk.Tk()
    app = TumorSimulatorApp(root)
    root.mainloop()
