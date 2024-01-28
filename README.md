# Tumor Simulator

The Tumor Simulator is a Python program designed for simulating the growth, death, and regeneration of tumor cells in a grid-based environment. The simulation is presented through a Graphical User Interface (GUI) using Tkinter, and visualized using Matplotlib.

## Features

- **Simulation Tab**: Displays the tumor grid and animates the simulation.
- **Controls Tab**: Allows users to adjust simulation parameters such as speed, growth, death, and regeneration probabilities.
- **Monte Carlo Simulation**: Run multiple simulations and generate visualizations for each run.
- **Graphs and Analysis**: Real-time and post-simulation graphs, including a summary page with mean and standard deviation.
- **Analytical Solution**: Compute and display an analytical solution using Euler's method for tumor growth.

## Getting Started

1. **Dependencies**: Ensure you have the necessary dependencies installed. You can install them using:

   ```bash
   pip install matplotlib numpy reportlab
   ```

2. **Run the Program**: Execute the script `tumor_simulator.py` to launch the Tumor Simulator GUI.

   ```bash
   python Cellular_Automata_Tumor_new.py
   ```

3. **Interact with the GUI**: Adjust simulation parameters, start the simulation, and explore the features.

## Usage

- **Simulation Controls**: Adjust parameters using sliders (speed, growth, death, living cell death, and regeneration probabilities).
- **Start Simulation**: Begin the simulation with the selected parameters.
- **Show Graphs**: Display graphs showing tumor cell counts during and after the simulation.
- **Run Monte Carlo Simulation**: Execute multiple simulations, visualize the results, and generate a summary PDF.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- The program utilizes Matplotlib and Tkinter for visualization and GUI components.
- Inspired by cellular automaton models for tumor growth.

Feel free to explore and customize the Tumor Simulator for your research or educational purposes. If you encounter any issues or have suggestions, please open an issue in the repository.
