# Sensors Evaluation with NSGA-II

This repository provides a simulation and visualization framework for evaluating sensor placement and coverage using a multi-objective genetic algorithm (NSGA-II). The code models sensors with a field of view, zones to be covered, and evolves sensor configurations to maximize coverage and minimize overlap.

## Features

- **Sensor and Zone Modeling:** Models sensors with adjustable position, angle, and range, and defines zones to be covered.
- **Genetic Algorithm (NSGA-II):** Evolves sensor placements using crossover, mutation, and non-dominated sorting to optimize for coverage and minimal overlap.
- **Visualization:** Plots sensor fields, zones, and Pareto frontiers using Matplotlib and Shapely.
- **GIF Creation:** Generates GIFs of the evolution process and Pareto frontiers.

## Requirements

- Python 3.8+
- [matplotlib](https://matplotlib.org/)
- [shapely](https://shapely.readthedocs.io/)
- [Pillow](https://python-pillow.org/)
- [numpy](https://numpy.org/)

Install dependencies with:

```bash
pip install matplotlib shapely Pillow numpy
```

## Usage

Run the main script:

```bash
python Objects.py
```

This will:

1. Initialize a set of sensors and zones.
2. Evolve the sensor placements over 100 generations.
3. Display and optionally save visualizations of the sensor coverage and Pareto front.

### Customization

- **Sensor Parameters:** Adjust the number, position, range, and angle of sensors in the `main()` function.
- **Zones:** Modify or add zones to be covered in the `main()` function.
- **Population Size & Generations:** Change `population_size` and the number of generations in `main()`.

### Output

- **Plots:** Visualizes sensor coverage and Pareto frontiers.
- **GIFs:** If enabled in `start_evolution`, saves GIFs of the evolution process and Pareto frontiers in the `evolution` and `pareto_frontier` folders.

## File Structure

- `Objects.py` — Main code for sensor modeling, genetic algorithm, and visualization.

## Example

The default configuration creates three sensors and three zones, then evolves sensor placements to maximize coverage and minimize overlap.

## License

MIT License 