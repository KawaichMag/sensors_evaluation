# Sensor Placement Optimization for UAV Platforms

This repository implements a multi-objective optimization framework for sensor placement on unmanned aerial vehicle (UAV) platforms. The system employs genetic algorithms with non-dominated sorting techniques to optimize sensor configurations for coverage efficiency and overlap minimization.

## Core Methodology

### Multi-Objective Optimization
The framework implements non-dominated sorting algorithms to identify Pareto-optimal solutions across multiple conflicting objectives. The optimization process balances sensor angle density against overlap reduction, generating diverse solution sets that represent trade-offs between these objectives.

### Genetic Operations
- **Mutation**: Sensor configurations undergo stochastic mutations using normal distribution sampling for both translational movement and rotational adjustments. Position mutations sample from a normal distribution with zero mean and unit variance, while rotational mutations use a normal distribution with zero mean and π/6 standard deviation.
- **Crossover**: Uniform crossover mechanism with 50% probability swapping for each sensor position between parent configurations.
- **Selection**: Non-dominated sorting determines population fitness, with selection based on front ranking and diversity preservation.

### Fitness Evaluation
The system evaluates solutions based on two primary metrics:
1. **Angle Density**: Measures the alignment of sensor orientations relative to the geometric center of the sensor array
2. **Overlap Coefficient**: Quantifies the intersection area between sensor coverage zones normalized by their union

## UAV Platform Models

The repository includes pre-configured sensor arrangements for multiple commercial UAV platforms:

- **DJI M300 RTK**: 8-sensor configuration on 670×810mm platform with 75°/65° field-of-view sensors
- **DJI M30**: 8-sensor configuration on 215×365mm platform with 65° field-of-view sensors  
- **Autel EVO II Pro v3**: 8-sensor configuration on 130×230mm platform with 60°/65° field-of-view sensors
- **DJI Mavic 3 Pro**: 8-sensor configuration on 98×230mm platform with 90° field-of-view sensors
- **EVO Nano**: 4-sensor configuration on 94×142mm platform with 40° field-of-view sensors

Each platform configuration includes realistic dimensional constraints and sensor specifications based on physical platform limitations.

## System Architecture

### Core Components
- `genetic_algorithm/`: Implementation of evolutionary algorithms and optimization routines
  - `evolution.py`: Main evolutionary loop with crossover, mutation, and selection operations
  - `fronts.py`: Non-dominated sorting and Pareto front identification algorithms
  - `metrics.py`: Fitness function definitions and objective evaluations
- `objects/`: Domain object definitions and geometric modeling
  - `Objects.py`: Sensor and zone geometric representations with polygon-based coverage modeling
  - `Types.py`: Type annotations for genetic algorithm data structures
- `utils/`: Visualization and analysis utilities
  - `drawing.py`: Matplotlib-based visualization for sensor arrangements and Pareto frontiers
  - `analysis.py`: Solution analysis and data extraction routines

### Computational Features
- Parallel population evaluation with configurable population sizes
- Adaptive constraint handling for platform boundary enforcement  
- Dynamic visualization generation including evolutionary progress tracking
- Automated GIF creation for temporal analysis of optimization convergence
- Pareto frontier analysis with solution archiving capabilities

## Requirements

- Python 3.13+
- matplotlib 3.10.3+
- shapely 2.1.1+
- pillow 11.3.0+

Install dependencies using:

```powershell
pip install matplotlib shapely pillow
```

## Usage

Execute platform-specific optimization:

```powershell
python dji_m30.py       # DJI M30 platform optimization
python m300rtk.py       # DJI M300 RTK platform optimization  
python mavic3pro.py     # DJI Mavic 3 Pro platform optimization
python autel_evo_ll-pro_v3.py  # Autel EVO II Pro v3 platform optimization
python evo_nano.py      # EVO Nano platform optimization
```

For custom configurations, modify the main.py file with desired sensor parameters and platform constraints.

### Configuration Parameters
- Population size: Typically 200-300 individuals for adequate diversity
- Generation count: 100-150 generations for convergence
- Mutation constraints: Platform-specific boundary enforcement
- Visualization options: Evolution tracking, Pareto frontier analysis, GIF generation

### Output Generation
The system produces:
- Static visualizations of optimal sensor configurations
- Pareto frontier plots showing objective trade-offs
- Evolutionary progress animations (optional)
- Solution analysis files in the `analysis/` directory
- Experimental data in the `experiments/` directory structure

## File Structure

```
├── genetic_algorithm/          # Optimization algorithm implementation
├── objects/                    # Domain object definitions  
├── utils/                      # Visualization and analysis tools
├── experiments/                # Generated experimental results
├── analysis/                   # Solution analysis outputs
├── [platform].py             # Platform-specific configurations
├── main.py                    # Generic optimization framework
└── pyproject.toml             # Project dependencies
```

## Research Applications

This framework supports research in autonomous systems, sensor network optimization, and multi-objective evolutionary computation. The modular architecture enables extension to additional UAV platforms and alternative optimization objectives while maintaining computational efficiency and solution quality. 