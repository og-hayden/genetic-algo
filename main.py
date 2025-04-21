"""
Genetic Algorithm for Scheduling Optimization
Author: Hayden Smith
Date: 2025-04-16

This program implements a genetic algorithm to optimize room, time, and facilitator
assignments for the Sophisticated Learning Association (SLA) activities.
"""

# NOTE: To run this program a super slick, totally legit (too legit to quit, I've heard it said) GUI, run `python main.py --gui`

import random
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import sys

# Import models and constants from models.py
from models import (
    TimeSlot, Room, Facilitator, Activity, ActivityAssignment, Schedule,
    POPULATION_SIZE, MAX_GENERATIONS, MUTATION_RATE, MIN_GENERATIONS, IMPROVEMENT_THRESHOLD
)

# Import other modules
from fitness import FitnessEvaluator
from genetic_ops import (
    generate_next_generation, 
    adaptive_mutation_rate, parallel_evaluate_fitness
)
from data import load_test_data


class GeneticAlgorithm:
    """Implements the genetic algorithm for schedule optimization."""
    
    def __init__(self, 
                 activities: list[Activity], 
                 rooms: list[Room], 
                 facilitators: list[Facilitator],
                 time_slots: list[TimeSlot],
                 population_size: int = POPULATION_SIZE,
                 mutation_rate: float = MUTATION_RATE):
        self.activities = activities
        self.rooms = rooms
        self.facilitators = facilitators
        self.time_slots = time_slots
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.fitness_evaluator = FitnessEvaluator()
        self.population: list[Schedule] = []
        self.best_fitness_history: list[float] = []
        self.avg_fitness_history: list[float] = []
    
    def initialize_population(self) -> None:
        """Generate initial random population of schedules."""
        print(f"Generating initial population of {self.population_size} schedules...")
        
        for _ in tqdm(range(self.population_size)):
            schedule = Schedule()
            
            # Create random assignments for each activity
            for activity in self.activities:
                # Randomly select room, time, and facilitator
                room = random.choice(self.rooms)
                time_slot = random.choice(self.time_slots)
                facilitator = random.choice(self.facilitators)
                
                # Create assignment
                assignment = ActivityAssignment(
                    activity=activity,
                    room=room,
                    time_slot=time_slot,
                    facilitator=facilitator
                )
                
                # Add to schedule
                schedule.assignments.append(assignment)
            
            # Add to population
            self.population.append(schedule)
        
        # Evaluate fitness for initial population
        print("Evaluating initial population fitness...")
        parallel_evaluate_fitness(self.population, self.fitness_evaluator)
    
    def evolve(self) -> None:
        """Evolve the population by one generation."""
        # Calculate adaptive mutation rate based on recent fitness history
        if len(self.avg_fitness_history) > 15:
            adaptive_rate = adaptive_mutation_rate(
                len(self.avg_fitness_history),
                self.avg_fitness_history
            )
        else:
            adaptive_rate = self.mutation_rate
        
        # Generate next generation
        self.population = generate_next_generation(
            self.population,
            self.rooms,
            self.facilitators,
            self.time_slots,
            elitism_count=5,
            mutation_rate=adaptive_rate
        )
        
        # Evaluate fitness for new population
        parallel_evaluate_fitness(self.population, self.fitness_evaluator)
    
    def run(self) -> tuple[Schedule, list[float], list[float]]:
        """Run the genetic algorithm until termination condition is met."""
        print("Starting genetic algorithm evolution...")
        generation = 0
        
        with tqdm(total=MAX_GENERATIONS) as pbar:
            while generation < MAX_GENERATIONS:
                self.evolve()
                generation += 1
                
                # Calculate fitness statistics for this generation
                current_fitnesses = [schedule.fitness for schedule in self.population]
                best_fitness = max(current_fitnesses)
                avg_fitness = sum(current_fitnesses) / len(current_fitnesses)
                
                self.best_fitness_history.append(best_fitness)
                self.avg_fitness_history.append(avg_fitness)
                
                # Update progress bar
                pbar.update(1)
                pbar.set_description(f"Gen {generation}: Best={best_fitness:.4f}, Avg={avg_fitness:.4f}")
                
                # Check termination condition after MIN_GENERATIONS
                if generation > MIN_GENERATIONS:
                    improvement = (avg_fitness - self.avg_fitness_history[generation - 100]) / abs(self.avg_fitness_history[generation - 100])
                    if improvement < IMPROVEMENT_THRESHOLD:
                        print(f"Terminating: improvement {improvement:.4f} < threshold {IMPROVEMENT_THRESHOLD}")
                        break
        
        # Return best schedule and fitness history
        best_schedule = max(self.population, key=lambda s: s.fitness)
        return best_schedule, self.best_fitness_history, self.avg_fitness_history


def create_test_data() -> tuple[list[Activity], list[Room], list[Facilitator], list[TimeSlot]]:
    """Create test data for activities, rooms, facilitators, and time slots."""
    # Use the data module to load test data
    return load_test_data()


def save_schedule(schedule: Schedule, filename: str) -> None:
    """Save the schedule to a file."""
    with open(filename, 'w') as f:
        f.write(f"Schedule Fitness: {schedule.fitness:.4f}\n\n")
        
        # Group assignments by time slot for better readability
        by_time = {}
        for assignment in schedule.assignments:
            time_str = str(assignment.time_slot)
            if time_str not in by_time:
                by_time[time_str] = []
            by_time[time_str].append(assignment)
        
        # Write time-organized schedule
        for time_str, assignments in sorted(by_time.items()):
            f.write(f"Time: {time_str}\n")
            f.write("-" * 80 + "\n")
            for assignment in sorted(assignments, key=lambda a: str(a.activity)):
                f.write(f"  {assignment.activity} -> Room: {assignment.room}, Facilitator: {assignment.facilitator}\n")
            f.write("\n")


def plot_fitness_history(best_fitness_history: list[float], avg_fitness_history: list[float], filename: str) -> None:
    """Plot the fitness history and save to a file."""
    plt.figure(figsize=(12, 6))
    plt.plot(best_fitness_history, label='Best Fitness')
    plt.plot(avg_fitness_history, label='Average Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness Evolution')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


def main() -> None:
    """Main function to run the genetic algorithm for schedule optimization."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Genetic Algorithm for Schedule Optimization')
    parser.add_argument('--gui', action='store_true', help='Launch the GUI visualization interface')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode for detailed logging')
    args = parser.parse_args()
    
    # Enable debug logging if requested
    if args.debug:
        import logging
        logging.basicConfig(level=logging.DEBUG, 
                           format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # If GUI mode is requested, launch the visualization
    if args.gui:
        try:
            from visualization import main as run_visualization
            print("Starting visualization interface...")
            run_visualization()
            return
        except ImportError as e:
            print(f"Error loading visualization module: {e}")
            print("Make sure PyQt5 and matplotlib are installed:")
            print("pip install PyQt5 matplotlib")
            sys.exit(1)
        except Exception as e:
            import traceback
            print(f"Error initializing visualization: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    # Otherwise run the command-line version
    print("Starting Genetic Algorithm for Schedule Optimization")
    start_time = time.time()
    
    # Create test data
    activities, rooms, facilitators, time_slots = create_test_data()
    
    # Print summary of data
    print(f"Loaded {len(activities)} activities, {len(rooms)} rooms, {len(facilitators)} facilitators, and {len(time_slots)} time slots")
    
    # Initialize and run genetic algorithm
    ga = GeneticAlgorithm(activities, rooms, facilitators, time_slots)
    ga.initialize_population()
    best_schedule, best_fitness_history, avg_fitness_history = ga.run()
    
    # Output results
    save_schedule(best_schedule, "best_schedule.txt")
    plot_fitness_history(best_fitness_history, avg_fitness_history, "fitness_history.png")
    
    elapsed_time = time.time() - start_time
    print(f"Optimization completed in {elapsed_time:.2f} seconds")
    print(f"Best schedule fitness: {best_schedule.fitness:.4f}")
    print(f"Schedule saved to best_schedule.txt")
    print(f"Fitness history plot saved to fitness_history.png")


if __name__ == "__main__":
    main() 