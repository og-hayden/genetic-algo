# Genetic Algorithm Scheduler

A Python implementation of a genetic algorithm to optimize the scheduling of rooms, time slots, and facilitators for the Sophisticated Learning Association (SLA) activities.

## Features

- Type-safe implementation with comprehensive documentation
- Parallel processing for fitness evaluation
- Visualization of fitness improvement over generations
- Adaptive mutation rate to balance exploration and exploitation

## Requirements

- Python 3.8+
- NumPy
- Matplotlib
- tqdm

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd genetic-algo
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

- `main.py`: Main entry point and core data structures
- `fitness.py`: Comprehensive fitness evaluation
- `genetic_ops.py`: Genetic algorithm operations (selection, crossover, mutation)
- `data.py`: Test data for activities, rooms, facilitators, and time slots
- `plan.md`: Implementation plan and design decisions

## Usage

Run the genetic algorithm with the default parameters:

```bash
python main.py
```

This will:
1. Generate an initial population of 500 random schedules
2. Evolve the population for at least 100 generations
3. Continue until improvement is less than 1%
4. Save the best schedule to `best_schedule.txt`
5. Generate a plot of fitness history in `fitness_history.png`

## Configuration

You can modify the following constants in `main.py`:

- `POPULATION_SIZE`: Number of schedules in each generation (default: 500)
- `MAX_GENERATIONS`: Maximum number of generations to run (default: 1000)
- `MUTATION_RATE`: Probability of mutation (default: 0.01)
- `MIN_GENERATIONS`: Minimum number of generations before checking improvement (default: 100)
- `IMPROVEMENT_THRESHOLD`: Termination condition threshold (default: 0.01)

## Output

The algorithm produces two outputs:

1. `best_schedule.txt`: A text file containing the best schedule found, organized by time slot
2. `fitness_history.png`: A plot showing the evolution of the best and average fitness over generations

## Performance Considerations

- The algorithm uses parallel processing for fitness evaluation
- The algorithm adapts the mutation rate based on recent fitness improvements
- Elitism preserves the best solutions across generations

## License

This project is licensed under the MIT License - see the LICENSE file for details. 