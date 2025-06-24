<<<<<<< HEAD
# paraller-and-distributed-computing-
 This project meticulously compares traditional serial processing against the power of local parallel computing using Python's `multiprocessing` module.  The application includes live performance metrics and a clear visual comparison plot, making it an excellent tool for understanding scalability and efficiency gains on multi-core systems. 
=======
# Traffic Flow Simulation (Serial & Parallel Processing)

This repository hosts a Python-based traffic flow simulation system designed to explore and demonstrate the performance benefits of parallel computing on local multi-core processors, comparing it against a traditional serial execution.

The simulation features a dynamic road network, realistic vehicle movement with basic routing, and traffic light control. It provides a comprehensive **Tkinter-based Graphical User Interface (GUI)** for real-time visualization and performance monitoring.

## Key Features:

* **City Zone Modeling:** Simulates traffic flow across multiple interconnected zones (e.g., North, South, East, West).
* **Vehicle Dynamics:** Models individual vehicle movement, including speed, position, and basic routing from origin to destination zones. Includes dynamic turning at intersections for more complex flow.
* **Traffic Light Control:** Basic traffic light mechanisms to manage flow at intersections, with adjustable cycle times.
* **Two Execution Modes:**
    * **Serial Mode:** A single-threaded baseline simulation. The entire simulation logic runs sequentially in one thread.
    * **Parallel Mode:** Leverages Python's `multiprocessing` module to run multiple independent simulation instances concurrently across available CPU cores. The GUI then aggregates/selects data from these parallel instances for a combined visual representation.
* **Tkinter GUI:** A desktop application providing:
    * Real-time visualization of colored vehicles moving on a detailed road network.
    * Dynamic traffic light state indicators at intersections.
    * Real-time information panels displaying zone-specific metrics (vehicle counts, congestion, avg. speed) and overall simulation performance.
    * A built-in `matplotlib` plot for direct visual comparison of serial versus parallel execution times and speedup factors.
* **Performance Analysis:** Designed for benchmarking the efficiency gains achieved through parallelization on multi-core systems.

## Technologies Used:

* **Python 3.x**
* **`tkinter`:** Standard Python library for creating the desktop graphical user interface.
* **`multiprocessing`:** Python module for spawning processes and managing inter-process communication (Queues, Values).
* **`matplotlib`:** Used for generating and embedding performance comparison plots within the Tkinter GUI.
* **`dataclasses`:** For defining structured data models for simulation entities (Point, Road, Vehicle).

## Getting Started:

To run the simulation locally:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```
    *(Remember to replace `your-username/your-repo-name` with your actual GitHub details)*

2.  **Install dependencies:**
    It's recommended to use a virtual environment.
    ```bash
    python -m venv venv
    .\venv\Scripts\activate  # On Windows
    source venv/bin/activate # On macOS/Linux
    ```
    Then install the required packages:
    ```bash
    pip install matplotlib Pillow
    ```
    *(Note: `tkinter` and `multiprocessing` are built-in Python modules and do not require `pip` installation.)*

3.  **Run the simulation:**
    ```bash
    python your_main_simulation_script_name.py
    ```
    *(Replace `your_main_simulation_script_name.py` with the actual name of the Python file containing the `TrafficSimulationGUI` class and the `if __name__ == "__main__":` block that starts the Tkinter application.)*

    This will launch the Tkinter GUI. From there, you can select between "Serial Processing" and "Parallel Processing" modes and start the simulation. To view the performance comparison, ensure you run both modes at least once, then click "Performance Comparison."

## Folder Structure (Example - adjust to your actual structure):
>>>>>>> af03c11 (Initial commit: Added serial and parallel traffic simulation with Tkinter GUI)
