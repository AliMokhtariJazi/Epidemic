# Epidemic Model Analysis

This repository contains the code and results for an epidemic model analysis assignment. The project includes simulations of epidemic dynamics using differential equations and network models.

## Repository Contents

- `part_one.py`: Python script for solving the differential equation model (Part A of the assignment)
- `part_two.py`: Python script for the network-based epidemic model (Part B of the assignment)
- `jacobian_analysis.pdf`: Mathematical analysis of the Jacobian matrix for stability analysis
- `animations/`: Folder containing epidemic spread animations
  - `part_a_animation.gif`: Animation of the differential equation model results
  - `part_b_animation.gif`: Animation of the network model epidemic spread
- `plots/`: Folder containing all generated plots
- `requirements.txt`: List of Python dependencies

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/epidemic-model-analysis.git
cd epidemic-model-analysis
```

### 2. Create a Virtual Environment

It's recommended to use a virtual environment to avoid conflicts with other Python projects.

For Unix or MacOS:
```bash
python3 -m venv env
source env/bin/activate
```

For Windows:
```bash
python -m venv env
.\env\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Running the Simulations

### Part One: Differential Equation Model

To run the simulation for the differential equation model:

```bash
python part_one.py
```

This will generate plots in the `plots/` directory and save the animation as `animations/part_a_animation.gif`.

### Part Two: Network Model

To run the network-based epidemic simulation:

```bash
python part_two.py
```

This will generate network plots in the `plots/` directory and save the animation as `animations/part_b_animation.gif`.

## Viewing Results

- All generated plots can be found in the `plots/` directory.
- Animations are saved in the `animations/` directory.
- For the mathematical analysis of the Jacobian matrix, refer to `jacobian_analysis.pdf`.

## Dependencies

The main dependencies for this project are:
- NumPy
- SciPy
- Matplotlib
- NetworkX

A full list of dependencies with version numbers can be found in `requirements.txt`.

## Notes

- Make sure you have sufficient disk space for generating and saving plots and animations.
- The simulations may take some time to run, especially for the network model with a large number of nodes.

If you encounter any issues or have questions, please open an issue in this repository.
