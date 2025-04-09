# Genetic Algorithm Scheduling - Implementation Plan

## Step 1: Data Modeling & Setup
- Create type-safe classes for all major entities:
  - `Activity`: name, expected enrollment, preferred/other facilitators
  - `Room`: name, capacity, building
  - `Facilitator`: name, specialties
  - `TimeSlot`: hour (assuming standard MWF slots)
  - `Schedule`: collection of `ActivityAssignment` objects
  - `ActivityAssignment`: links an activity to room, time, and facilitator
- Define mock data for activities, rooms, and facilitators
- Setup GPU acceleration with PyTorch or CuPy for parallel fitness calculations
- Implement utility functions for time slot calculations (e.g., consecutive slots, hours apart)

## Step 2: Generate Initial Population
- Create a `Population` class to manage schedules
- Implement random schedule generation ensuring:
  - Each activity has an assigned room, time, and facilitator
  - Special handling for SLA 101 and SLA 191 sections (A/B)
- Generate 500+ random schedules to form initial population
- Implement data validation to ensure structural integrity

## Step 3: Fitness Function Implementation
- Create a comprehensive `FitnessEvaluator` class with the following components:
  - Room conflict detection
  - Room size appropriateness calculation
  - Facilitator preference scoring
  - Facilitator workload analysis
  - Special activity relationship scoring (SLA 101/191)
  - Building location considerations for consecutive activities
- Use vectorized operations where possible (NumPy/CuPy) for GPU acceleration
- Implement caching mechanisms to avoid redundant calculations
- Create visualization tools to analyze fitness components

## Step 4: Genetic Algorithm Core
- Implement selection mechanism using softmax normalization
- Design crossover operation:
  - Strategy for combining parent schedules
  - Ensuring valid offspring schedules are produced
- Implement mutation with configurable rate (starting at 0.01)
  - Room mutations
  - Time slot mutations
  - Facilitator mutations
- Create next generation selection logic
- Implement parallel processing for fitness evaluation and selection

## Step 5: Execution, Monitoring, and Optimization
- Run the algorithm for at least 100 generations
- Implement termination condition (<1% improvement over G-100)
- Add logging and progress monitoring
- Create automated parameter tuning for mutation rate
- Output best schedule to file with clear formatting
- Implement visualization of fitness improvement over generations
- Add profiling and optimization capabilities
- Generate final report with metrics and analysis 