#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genetic Operations Module for the Scheduling Problem
This module contains implementations of selection, crossover, and mutation operations.
"""

import random
import copy
from typing import List, Tuple, Dict, Set, Any, Optional, Union, Callable
import numpy as np

# Import from models module
from models import (
    Schedule, Activity, Room, Facilitator, TimeSlot, 
    ActivityAssignment, MUTATION_RATE
)


def softmax(fitness_values: List[float], temperature: float = 1.0) -> np.ndarray:
    """
    Compute softmax values for each fitness in the list.
    
    Args:
        fitness_values: List of fitness values
        temperature: Temperature parameter for softmax (higher = more uniform)
        
    Returns:
        Normalized probabilities for each fitness value
    """
    x = np.array(fitness_values) / temperature
    # Shift for numerical stability
    x = x - np.max(x)
    e_x = np.exp(x)
    return e_x / np.sum(e_x)


def select_parents(population: List[Schedule], num_parents: int = 2) -> List[Schedule]:
    """
    Select parents from the population using softmax selection.
    
    Args:
        population: List of schedules
        num_parents: Number of parents to select
        
    Returns:
        List of selected parent schedules
    """
    # Extract fitness values
    fitness_values = [schedule.fitness for schedule in population]
    
    # Apply softmax normalization to get selection probabilities
    probabilities = softmax(fitness_values)
    
    # Select parents based on probabilities
    selected_indices = np.random.choice(
        len(population), 
        size=num_parents, 
        p=probabilities,
        replace=False  # Without replacement to get different parents
    )
    
    return [population[i] for i in selected_indices]


def crossover(parent1: Schedule, parent2: Schedule) -> Schedule:
    """
    Create a new schedule by combining elements from two parent schedules.
    
    This crossover randomly selects assignments from either parent1 or parent2
    for each activity, ensuring we don't have duplicate assignments.
    
    Args:
        parent1: First parent schedule
        parent2: Second parent schedule
        
    Returns:
        A new offspring schedule
    """
    # Create a new empty schedule
    offspring = Schedule()
    
    # Create a dictionary to look up assignments by activity name and section
    p1_assignments = {}
    for assignment in parent1.assignments:
        activity = assignment.activity
        key = (activity.name, activity.section)
        p1_assignments[key] = assignment
    
    p2_assignments = {}
    for assignment in parent2.assignments:
        activity = assignment.activity
        key = (activity.name, activity.section)
        p2_assignments[key] = assignment
    
    # For each activity, choose an assignment from either parent
    for key in p1_assignments.keys():
        # Get assignments from both parents
        p1_assignment = p1_assignments[key]
        p2_assignment = p2_assignments[key]
        
        # Randomly choose one of the parents' assignments
        chosen_assignment = random.choice([p1_assignment, p2_assignment])
        
        # Add a deep copy to prevent modification of original
        offspring.assignments.append(copy.deepcopy(chosen_assignment))
    
    # We don't evaluate fitness here - that will be done by the main algorithm
    return offspring


def mutate(schedule: Schedule, 
           available_rooms: List[Room],
           available_facilitators: List[Facilitator], 
           available_time_slots: List[TimeSlot],
           mutation_rate: float = MUTATION_RATE) -> None:
    """
    Apply random mutations to a schedule with probability based on mutation_rate.
    
    This function mutates the schedule in-place by randomly changing room, time,
    or facilitator assignments based on the mutation rate.
    
    Args:
        schedule: The schedule to mutate
        available_rooms: List of all possible rooms
        available_facilitators: List of all possible facilitators
        available_time_slots: List of all possible time slots
        mutation_rate: Probability of mutating each element
        
    Returns:
        None (mutations are applied in-place)
    """
    for assignment in schedule.assignments:
        # Determine if this assignment should be mutated
        if random.random() < mutation_rate:
            # Randomly choose what aspect to mutate (room, time, or facilitator)
            mutation_type = random.choice(["room", "time", "facilitator"])
            
            if mutation_type == "room":
                # Mutate room assignment
                assignment.room = random.choice(available_rooms)
                
            elif mutation_type == "time":
                # Mutate time slot assignment
                assignment.time_slot = random.choice(available_time_slots)
                
            elif mutation_type == "facilitator":
                # Mutate facilitator assignment
                assignment.facilitator = random.choice(available_facilitators)


def tournament_selection(population: List[Schedule], tournament_size: int = 3) -> Schedule:
    """
    Alternative selection method using tournament selection.
    
    Args:
        population: List of schedules
        tournament_size: Number of schedules to include in each tournament
        
    Returns:
        The winner of the tournament (highest fitness)
    """
    # Randomly select schedules for the tournament
    tournament = random.sample(population, tournament_size)
    
    # Return the schedule with the highest fitness
    return max(tournament, key=lambda schedule: schedule.fitness)


def generate_next_generation(
    current_population: List[Schedule],
    available_rooms: List[Room],
    available_facilitators: List[Facilitator],
    available_time_slots: List[TimeSlot],
    elitism_count: int = 5,
    mutation_rate: float = MUTATION_RATE,
    use_tournament: bool = False
) -> List[Schedule]:
    """
    Generate the next generation of schedules using genetic operations.
    
    Args:
        current_population: The current generation of schedules
        available_rooms: List of all possible rooms
        available_facilitators: List of all possible facilitators
        available_time_slots: List of all possible time slots
        elitism_count: Number of top schedules to preserve unchanged
        mutation_rate: Probability of mutation
        use_tournament: Whether to use tournament selection instead of softmax
        
    Returns:
        The next generation of schedules
    """
    population_size = len(current_population)
    next_generation = []
    
    # Sort population by fitness (descending)
    sorted_population = sorted(current_population, key=lambda s: s.fitness, reverse=True)
    
    # Elitism: Keep top schedules unchanged
    for i in range(elitism_count):
        # Add deep copy to prevent modification
        next_generation.append(copy.deepcopy(sorted_population[i]))
    
    # Generate remaining schedules through selection, crossover, and mutation
    while len(next_generation) < population_size:
        # Select parents
        if use_tournament:
            parent1 = tournament_selection(current_population)
            parent2 = tournament_selection(current_population)
        else:
            parent1, parent2 = select_parents(current_population, num_parents=2)
        
        # Create offspring through crossover
        offspring = crossover(parent1, parent2)
        
        # Mutate offspring
        mutate(offspring, available_rooms, available_facilitators, available_time_slots, mutation_rate)
        
        # Add to next generation
        next_generation.append(offspring)
    
    # Ensure we don't have more than population_size
    return next_generation[:population_size]


def adaptive_mutation_rate(generation: int, 
                          avg_fitness_history: List[float],
                          min_rate: float = 0.001,
                          max_rate: float = 0.05,
                          patience: int = 15) -> float:
    """
    Adaptively adjust mutation rate based on recent fitness improvements.
    
    If fitness is not improving over recent generations, increase mutation rate
    to escape local optima. If fitness is improving well, reduce mutation rate
    for fine-tuning.
    
    Args:
        generation: Current generation number
        avg_fitness_history: History of average fitness values
        min_rate: Minimum mutation rate
        max_rate: Maximum mutation rate
        patience: Number of generations to consider for improvement
        
    Returns:
        Adjusted mutation rate
    """
    if generation <= patience:
        return max_rate  # Start with high mutation rate
    
    # Calculate recent improvement
    recent_improvement = avg_fitness_history[-1] - avg_fitness_history[-patience]
    
    if recent_improvement <= 0:
        # No improvement, increase mutation rate to escape local optima
        return max_rate
    elif recent_improvement < 0.01:
        # Small improvement, slightly increase mutation
        return min(max_rate, MUTATION_RATE * 1.5)
    else:
        # Good improvement, reduce mutation rate for fine-tuning
        return max(min_rate, MUTATION_RATE * 0.5)


# Parallel processing functions for fitness evaluation
def parallel_evaluate_fitness(schedules: List[Schedule], 
                             evaluator: Any,
                             num_processes: Optional[int] = None) -> List[float]:
    """
    Evaluate fitness of multiple schedules in parallel.
    
    Args:
        schedules: List of schedules to evaluate
        evaluator: The fitness evaluator object
        num_processes: Number of processes to use (None = auto)
        
    Returns:
        List of fitness values
    """
    # Use a simpler approach without multiprocessing to avoid pickling issues
    fitness_values = []
    for schedule in schedules:
        fitness = evaluator.evaluate(schedule)
        schedule.fitness = fitness
        fitness_values.append(fitness)
    
    return fitness_values 