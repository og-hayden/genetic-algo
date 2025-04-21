"""
Visualization Module for Genetic Algorithm Scheduler
This module provides a PyQt-based graphical user interface to visualize the genetic algorithm
and resulting schedules.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTabWidget, QTableWidget, QTableWidgetItem,
    QProgressBar, QSpinBox, QDoubleSpinBox, QGroupBox,
    QGridLayout, QSplitter, QHeaderView, QFileDialog, QMessageBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor

from models import (
    POPULATION_SIZE, MAX_GENERATIONS, MUTATION_RATE, MIN_GENERATIONS
)
from genetic_ops import generate_next_generation, parallel_evaluate_fitness
from data import load_test_data
from main import GeneticAlgorithm


class MatplotlibCanvas(FigureCanvasQTAgg):
    """Custom Matplotlib canvas for embedding in Qt interface."""
    
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MatplotlibCanvas, self).__init__(self.fig)
        self.setParent(parent)
        self.fig.tight_layout()


class GeneticAlgorithmThread(QThread):
    """Thread for running the genetic algorithm in the background."""
    
    # Signals for updating UI
    update_progress = pyqtSignal(int, float, float)
    update_best_schedule = pyqtSignal(object)
    finished_evolution = pyqtSignal(object, list, list)
    update_status = pyqtSignal(str)  # New signal for status updates
    
    def __init__(self, activities, rooms, facilitators, time_slots, 
                 population_size=POPULATION_SIZE, mutation_rate=MUTATION_RATE,
                 max_generations=MAX_GENERATIONS):
        super().__init__()
        self.activities = activities
        self.rooms = rooms
        self.facilitators = facilitators
        self.time_slots = time_slots
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations
        self.ga = None
        self.generation = 0
        self.stopped = False
    
    def run(self):
        """Run the genetic algorithm in a separate thread."""
        try:
            self.stopped = False
            
            # Create the GA instance
            self.update_status.emit("Creating genetic algorithm instance...")
            self.ga = GeneticAlgorithm(
                self.activities, 
                self.rooms, 
                self.facilitators, 
                self.time_slots,
                self.population_size,
                self.mutation_rate
            )
            
            # Initialize population with status updates
            self.update_status.emit("Initializing population. This may take a moment...")
            self.ga.initialize_population()
            
            # Get best schedule from initial population
            self.update_status.emit("Initial population created. Processing best schedule...")
            best_schedule = max(self.ga.population, key=lambda s: s.fitness)
            self.update_best_schedule.emit(best_schedule)
            
            # Start evolution process
            self.update_status.emit("Starting evolution process...")
            self.generation = 0
            
            current_fitnesses = [schedule.fitness for schedule in self.ga.population]
            best_fitness = max(current_fitnesses)
            avg_fitness = sum(current_fitnesses) / len(current_fitnesses)
            
            # Initialize fitness history with first generation
            self.ga.best_fitness_history = [best_fitness]
            self.ga.avg_fitness_history = [avg_fitness]
            
            # Update initial progress
            self.update_progress.emit(0, best_fitness, avg_fitness)
            
            while self.generation < self.max_generations and not self.stopped:
                self.ga.evolve()
                self.generation += 1
                
                # Calculate fitness statistics for this generation
                current_fitnesses = [schedule.fitness for schedule in self.ga.population]
                best_fitness = max(current_fitnesses)
                avg_fitness = sum(current_fitnesses) / len(current_fitnesses)
                
                self.ga.best_fitness_history.append(best_fitness)
                self.ga.avg_fitness_history.append(avg_fitness)
                
                # Update progress and best schedule
                self.update_progress.emit(self.generation, best_fitness, avg_fitness)
                
                # Get best schedule
                best_schedule = max(self.ga.population, key=lambda s: s.fitness)
                self.update_best_schedule.emit(best_schedule)
                
                # Check termination condition after MIN_GENERATIONS
                if self.generation > MIN_GENERATIONS:
                    improvement = (avg_fitness - self.ga.avg_fitness_history[self.generation - 100]) / abs(self.ga.avg_fitness_history[self.generation - 100])
                    if improvement < 0.01:  # IMPROVEMENT_THRESHOLD
                        self.update_status.emit(f"Terminating: improvement {improvement:.4f} < threshold 0.01")
                        break
            
            # Return best schedule and fitness history
            best_schedule = max(self.ga.population, key=lambda s: s.fitness)
            self.update_status.emit("Evolution completed successfully!")
            self.finished_evolution.emit(best_schedule, self.ga.best_fitness_history, self.ga.avg_fitness_history)
            
        except Exception as e:
            import traceback
            self.update_status.emit(f"Error in genetic algorithm: {str(e)}")
            traceback.print_exc()
    
    def stop(self):
        """Stop the genetic algorithm thread."""
        self.stopped = True


class GridSearchThread(QThread):
    """Thread for running the grid search in the background."""
    
    # Signals for updating UI
    update_progress = pyqtSignal(int, int, str)
    update_result = pyqtSignal(str, float, float, int)  # Added elitism to result signal
    finished_search = pyqtSignal()
    
    def __init__(self, activities, rooms, facilitators, time_slots, 
                 param_combinations, generations_per_run):
        super().__init__()
        self.activities = activities
        self.rooms = rooms
        self.facilitators = facilitators
        self.time_slots = time_slots
        self.param_combinations = param_combinations
        self.generations_per_run = generations_per_run
        self.stopped = False
        
        # Track results
        self.results = []
    
    def run(self):
        """Run the grid search by testing each parameter combination."""
        try:
            total_combinations = len(self.param_combinations)
            
            for i, (pop_size, mut_rate, elit_count) in enumerate(self.param_combinations):
                if self.stopped:
                    break
                
                # Create parameter string
                param_str = f"Population Size: {pop_size}, Mutation Rate: {mut_rate:.4f}, Elitism: {elit_count}"
                
                # Update progress
                self.update_progress.emit(i, total_combinations, param_str)
                
                # Run the GA with these parameters
                ga = GeneticAlgorithm(
                    self.activities,
                    self.rooms,
                    self.facilitators,
                    self.time_slots,
                    pop_size,
                    mut_rate
                )
                
                # Initialize population
                ga.initialize_population()
                
                # Run for a fixed number of generations
                for gen in range(self.generations_per_run):
                    if self.stopped:
                        break
                    
                    # Use elitism count parameter in evolve
                    ga.population = generate_next_generation(
                        ga.population,
                        ga.rooms,
                        ga.facilitators,
                        ga.time_slots,
                        elitism_count=elit_count,
                        mutation_rate=mut_rate
                    )
                    
                    # Evaluate fitness
                    parallel_evaluate_fitness(ga.population, ga.fitness_evaluator)
                    
                    # Update progress message every 10 generations
                    if gen % 10 == 0:
                        progress_msg = f"{param_str} (Gen {gen}/{self.generations_per_run})"
                        self.update_progress.emit(i, total_combinations, progress_msg)
                
                if not self.stopped:
                    # Calculate fitness statistics
                    fitnesses = [schedule.fitness for schedule in ga.population]
                    avg_fitness = sum(fitnesses) / len(fitnesses)
                    best_fitness = max(fitnesses)
                    
                    # Store result
                    self.results.append((pop_size, mut_rate, elit_count, avg_fitness, best_fitness))
                    
                    # Emit result with elitism count included
                    self.update_result.emit(param_str, avg_fitness, best_fitness, elit_count)
            
            # Signal completion
            if not self.stopped:
                self.finished_search.emit()
                
        except Exception as e:
            import traceback
            print(f"Error in grid search: {str(e)}")
            traceback.print_exc()
    
    def stop(self):
        """Stop the grid search thread."""
        self.stopped = True


class ScheduleVisualizer(QMainWindow):
    """Main window for the genetic algorithm visualization."""
    
    def __init__(self):
        super().__init__()
        
        # Load test data
        self.activities, self.rooms, self.facilitators, self.time_slots = load_test_data()
        
        # Initialize UI
        self.setWindowTitle("Genetic Algorithm Schedule Visualizer")
        self.setGeometry(100, 100, 1200, 800)
        self.setup_ui()
        
        # Initialize GA thread
        self.ga_thread = None
        self.current_best_schedule = None
    
    def setup_ui(self):
        """Set up the main user interface."""
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Create tabs
        self.tabs = QTabWidget()
        self.main_tab = QWidget()
        self.stats_tab = QWidget()
        self.grid_search_tab = QWidget()
        
        self.tabs.addTab(self.main_tab, "Schedule & Controls")
        self.tabs.addTab(self.stats_tab, "Statistics")
        self.tabs.addTab(self.grid_search_tab, "Grid Search")
        
        # Setup tab contents
        self.setup_main_tab()
        self.setup_stats_tab()
        self.setup_grid_search_tab()
        
        # Add tabs to main layout
        main_layout.addWidget(self.tabs)
    
    def setup_main_tab(self):
        """Set up the main tab with both controls and schedule view."""
        # Main layout as a horizontal splitter
        main_layout = QHBoxLayout(self.main_tab)
        
        # Create left panel for controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Parameters group
        params_group = QGroupBox("Algorithm Parameters")
        params_layout = QGridLayout()
        
        # Population size
        params_layout.addWidget(QLabel("Population Size:"), 0, 0)
        self.population_size_spin = QSpinBox()
        self.population_size_spin.setRange(100, 2000)
        self.population_size_spin.setValue(POPULATION_SIZE)
        self.population_size_spin.setSingleStep(50)
        params_layout.addWidget(self.population_size_spin, 0, 1)
        
        # Mutation rate
        params_layout.addWidget(QLabel("Mutation Rate:"), 1, 0)
        self.mutation_rate_spin = QDoubleSpinBox()
        self.mutation_rate_spin.setRange(0.001, 0.1)
        self.mutation_rate_spin.setValue(MUTATION_RATE)
        self.mutation_rate_spin.setSingleStep(0.001)
        self.mutation_rate_spin.setDecimals(4)
        params_layout.addWidget(self.mutation_rate_spin, 1, 1)
        
        # Max generations
        params_layout.addWidget(QLabel("Max Generations:"), 2, 0)
        self.max_gen_spin = QSpinBox()
        self.max_gen_spin.setRange(100, 5000)
        self.max_gen_spin.setValue(MAX_GENERATIONS)
        self.max_gen_spin.setSingleStep(100)
        params_layout.addWidget(self.max_gen_spin, 2, 1)
        
        params_group.setLayout(params_layout)
        left_layout.addWidget(params_group)
        
        # Progress information
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout()
        
        # Add status label
        self.status_label = QLabel("Ready to start evolution")
        self.status_label.setWordWrap(True)
        font = self.status_label.font()
        font.setBold(True)
        self.status_label.setFont(font)
        progress_layout.addWidget(self.status_label)
        
        self.generation_label = QLabel("Generation: 0 / 0")
        progress_layout.addWidget(self.generation_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        
        self.fitness_label = QLabel("Best Fitness: 0.0000 | Avg Fitness: 0.0000")
        progress_layout.addWidget(self.fitness_label)
        
        progress_group.setLayout(progress_layout)
        left_layout.addWidget(progress_group)
        
        # Control buttons
        buttons_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Start Evolution")
        self.start_button.clicked.connect(self.start_evolution)
        buttons_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("Stop Evolution")
        self.stop_button.clicked.connect(self.stop_evolution)
        self.stop_button.setEnabled(False)
        buttons_layout.addWidget(self.stop_button)
        
        self.save_button = QPushButton("Save Best Schedule")
        self.save_button.clicked.connect(self.save_schedule)
        self.save_button.setEnabled(False)
        buttons_layout.addWidget(self.save_button)
        
        left_layout.addLayout(buttons_layout)
        
        # Summary information
        summary_group = QGroupBox("Data Summary")
        summary_layout = QVBoxLayout()
        
        summary_text = f"""
        Activities: {len(self.activities)}
        Rooms: {len(self.rooms)}
        Facilitators: {len(self.facilitators)}
        Time Slots: {len(self.time_slots)}
        """
        summary_label = QLabel(summary_text)
        summary_layout.addWidget(summary_label)
        
        summary_group.setLayout(summary_layout)
        left_layout.addWidget(summary_group)
        
        # Add stretch to push everything to the top
        left_layout.addStretch()
        
        # Create right panel for schedule
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Schedule view title
        schedule_label = QLabel("Schedule View")
        schedule_label.setAlignment(Qt.AlignCenter)
        font = schedule_label.font()
        font.setBold(True)
        font.setPointSize(12)
        schedule_label.setFont(font)
        right_layout.addWidget(schedule_label)
        
        # Create a grid representation of the schedule
        self.schedule_grid = QTableWidget()
        self.schedule_grid.setColumnCount(len(self.rooms) + 1)  # +1 for time slots
        self.schedule_grid.setRowCount(len(self.time_slots))
        
        # Set headers
        self.schedule_grid.setHorizontalHeaderItem(0, QTableWidgetItem("Time Slot"))
        for i, room in enumerate(self.rooms):
            self.schedule_grid.setHorizontalHeaderItem(i + 1, QTableWidgetItem(str(room)))
        
        # Set row labels (time slots)
        for i, time_slot in enumerate(self.time_slots):
            self.schedule_grid.setItem(i, 0, QTableWidgetItem(str(time_slot)))
        
        # Adjust column widths
        self.schedule_grid.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        # Add the table to the layout
        right_layout.addWidget(self.schedule_grid)
        
        # Add facilitator loads table
        facilitator_label = QLabel("Facilitator Workload")
        facilitator_label.setAlignment(Qt.AlignCenter)
        font = facilitator_label.font()
        font.setBold(True)
        facilitator_label.setFont(font)
        right_layout.addWidget(facilitator_label)
        
        self.facilitator_table = QTableWidget()
        self.facilitator_table.setColumnCount(2)
        self.facilitator_table.setRowCount(len(self.facilitators))
        self.facilitator_table.setHorizontalHeaderLabels(["Facilitator", "Activity Count"])
        
        # Initialize facilitator rows
        for i, facilitator in enumerate(self.facilitators):
            self.facilitator_table.setItem(i, 0, QTableWidgetItem(str(facilitator)))
            self.facilitator_table.setItem(i, 1, QTableWidgetItem("0"))
        
        self.facilitator_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        right_layout.addWidget(self.facilitator_table)
        
        # Create a splitter to allow resizing the panels
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([300, 900])  # Initial sizes
        
        # Add the splitter to the main layout
        main_layout.addWidget(splitter)
    
    def setup_stats_tab(self):
        """Set up the statistics visualization tab."""
        layout = QVBoxLayout(self.stats_tab)
        
        # Create the matplotlib canvas for fitness history
        self.fitness_canvas = MatplotlibCanvas(self.stats_tab, width=10, height=8)
        layout.addWidget(self.fitness_canvas)
        
        # Add a summary of key statistics
        stats_group = QGroupBox("Statistics Summary")
        stats_layout = QVBoxLayout()
        
        self.stats_label = QLabel("No evolution has been run yet.")
        stats_layout.addWidget(self.stats_label)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
    
    def setup_grid_search_tab(self):
        """Set up the grid search tab for hyperparameter tuning."""
        layout = QVBoxLayout(self.grid_search_tab)
        
        # Parameter ranges group
        param_group = QGroupBox("Parameter Ranges")
        param_layout = QGridLayout()
        
        # Population size range
        param_layout.addWidget(QLabel("Population Size:"), 0, 0)
        
        pop_size_min_layout = QHBoxLayout()
        pop_size_min_layout.addWidget(QLabel("Min:"))
        self.pop_size_min = QSpinBox()
        self.pop_size_min.setRange(50, 1000)
        self.pop_size_min.setValue(100)
        self.pop_size_min.setSingleStep(50)
        pop_size_min_layout.addWidget(self.pop_size_min)
        param_layout.addLayout(pop_size_min_layout, 0, 1)
        
        pop_size_max_layout = QHBoxLayout()
        pop_size_max_layout.addWidget(QLabel("Max:"))
        self.pop_size_max = QSpinBox()
        self.pop_size_max.setRange(100, 2000)
        self.pop_size_max.setValue(500)
        self.pop_size_max.setSingleStep(50)
        pop_size_max_layout.addWidget(self.pop_size_max)
        param_layout.addLayout(pop_size_max_layout, 0, 2)
        
        pop_size_step_layout = QHBoxLayout()
        pop_size_step_layout.addWidget(QLabel("Step:"))
        self.pop_size_step = QSpinBox()
        self.pop_size_step.setRange(50, 500)
        self.pop_size_step.setValue(100)
        self.pop_size_step.setSingleStep(50)
        pop_size_step_layout.addWidget(self.pop_size_step)
        param_layout.addLayout(pop_size_step_layout, 0, 3)
        
        # Mutation rate range
        param_layout.addWidget(QLabel("Mutation Rate:"), 1, 0)
        
        mut_rate_min_layout = QHBoxLayout()
        mut_rate_min_layout.addWidget(QLabel("Min:"))
        self.mut_rate_min = QDoubleSpinBox()
        self.mut_rate_min.setRange(0.001, 0.05)
        self.mut_rate_min.setValue(0.001)
        self.mut_rate_min.setSingleStep(0.001)
        self.mut_rate_min.setDecimals(4)
        mut_rate_min_layout.addWidget(self.mut_rate_min)
        param_layout.addLayout(mut_rate_min_layout, 1, 1)
        
        mut_rate_max_layout = QHBoxLayout()
        mut_rate_max_layout.addWidget(QLabel("Max:"))
        self.mut_rate_max = QDoubleSpinBox()
        self.mut_rate_max.setRange(0.005, 0.1)
        self.mut_rate_max.setValue(0.05)
        self.mut_rate_max.setSingleStep(0.005)
        self.mut_rate_max.setDecimals(4)
        mut_rate_max_layout.addWidget(self.mut_rate_max)
        param_layout.addLayout(mut_rate_max_layout, 1, 2)
        
        mut_rate_step_layout = QHBoxLayout()
        mut_rate_step_layout.addWidget(QLabel("Step:"))
        self.mut_rate_step = QDoubleSpinBox()
        self.mut_rate_step.setRange(0.001, 0.02)
        self.mut_rate_step.setValue(0.005)
        self.mut_rate_step.setSingleStep(0.001)
        self.mut_rate_step.setDecimals(4)
        mut_rate_step_layout.addWidget(self.mut_rate_step)
        param_layout.addLayout(mut_rate_step_layout, 1, 3)
        
        # Elitism count range (third parameter)
        param_layout.addWidget(QLabel("Elitism Count:"), 2, 0)
        
        elitism_min_layout = QHBoxLayout()
        elitism_min_layout.addWidget(QLabel("Min:"))
        self.elitism_min = QSpinBox()
        self.elitism_min.setRange(1, 10)
        self.elitism_min.setValue(2)
        self.elitism_min.setSingleStep(1)
        elitism_min_layout.addWidget(self.elitism_min)
        param_layout.addLayout(elitism_min_layout, 2, 1)
        
        elitism_max_layout = QHBoxLayout()
        elitism_max_layout.addWidget(QLabel("Max:"))
        self.elitism_max = QSpinBox()
        self.elitism_max.setRange(5, 25)
        self.elitism_max.setValue(10)
        self.elitism_max.setSingleStep(1)
        elitism_max_layout.addWidget(self.elitism_max)
        param_layout.addLayout(elitism_max_layout, 2, 2)
        
        elitism_step_layout = QHBoxLayout()
        elitism_step_layout.addWidget(QLabel("Step:"))
        self.elitism_step = QSpinBox()
        self.elitism_step.setRange(1, 5)
        self.elitism_step.setValue(2)
        self.elitism_step.setSingleStep(1)
        elitism_step_layout.addWidget(self.elitism_step)
        param_layout.addLayout(elitism_step_layout, 2, 3)
        
        # Max generations for each run
        param_layout.addWidget(QLabel("Generations per Run:"), 3, 0)
        self.gs_generations = QSpinBox()
        self.gs_generations.setRange(50, 300)
        self.gs_generations.setValue(100)
        self.gs_generations.setSingleStep(10)
        param_layout.addWidget(self.gs_generations, 3, 1)
        
        # Number of trials for each parameter combination
        param_layout.addWidget(QLabel("Trials per Combination:"), 4, 0)
        self.gs_trials = QSpinBox()
        self.gs_trials.setRange(1, 5)
        self.gs_trials.setValue(1)  # Changed to 1 for faster grid search
        param_layout.addWidget(self.gs_trials, 4, 1)
        
        param_group.setLayout(param_layout)
        layout.addWidget(param_group)
        
        # Split the rest of the interface into two columns
        rest_layout = QHBoxLayout()
        
        # Left column for information and controls
        left_column = QVBoxLayout()
        
        # Estimated runs and time
        info_group = QGroupBox("Grid Search Information")
        info_layout = QVBoxLayout()
        
        self.gs_runs_label = QLabel("Estimated runs: 0")
        info_layout.addWidget(self.gs_runs_label)
        
        self.gs_time_label = QLabel("Estimated time: 0 minutes")
        info_layout.addWidget(self.gs_time_label)
        
        # Connect signals to update the estimated runs and time
        self.pop_size_min.valueChanged.connect(self.update_grid_search_info)
        self.pop_size_max.valueChanged.connect(self.update_grid_search_info)
        self.pop_size_step.valueChanged.connect(self.update_grid_search_info)
        self.mut_rate_min.valueChanged.connect(self.update_grid_search_info)
        self.mut_rate_max.valueChanged.connect(self.update_grid_search_info)
        self.mut_rate_step.valueChanged.connect(self.update_grid_search_info)
        self.elitism_min.valueChanged.connect(self.update_grid_search_info)
        self.elitism_max.valueChanged.connect(self.update_grid_search_info)
        self.elitism_step.valueChanged.connect(self.update_grid_search_info)
        self.gs_trials.valueChanged.connect(self.update_grid_search_info)
        
        info_group.setLayout(info_layout)
        left_column.addWidget(info_group)
        
        # Results table
        results_group = QGroupBox("Grid Search Results")
        results_layout = QVBoxLayout()
        
        self.gs_results_table = QTableWidget()
        self.gs_results_table.setColumnCount(4)  # Added one more column for elitism
        self.gs_results_table.setHorizontalHeaderLabels(["Parameters", "Avg Fitness", "Best Fitness", "Elitism"])
        self.gs_results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        results_layout.addWidget(self.gs_results_table)
        
        results_group.setLayout(results_layout)
        left_column.addWidget(results_group)
        
        # Control buttons
        buttons_layout = QHBoxLayout()
        
        self.start_gs_button = QPushButton("Start Grid Search")
        self.start_gs_button.clicked.connect(self.start_grid_search)
        buttons_layout.addWidget(self.start_gs_button)
        
        self.stop_gs_button = QPushButton("Stop Grid Search")
        self.stop_gs_button.clicked.connect(self.stop_grid_search)
        self.stop_gs_button.setEnabled(False)
        buttons_layout.addWidget(self.stop_gs_button)
        
        self.use_best_button = QPushButton("Use Best Parameters")
        self.use_best_button.clicked.connect(self.use_best_parameters)
        self.use_best_button.setEnabled(False)
        buttons_layout.addWidget(self.use_best_button)
        
        left_column.addLayout(buttons_layout)
        
        # Progress bar for grid search
        self.gs_progress_bar = QProgressBar()
        self.gs_progress_bar.setRange(0, 100)
        self.gs_progress_bar.setValue(0)
        left_column.addWidget(self.gs_progress_bar)
        
        self.gs_status_label = QLabel("Ready to start grid search")
        left_column.addWidget(self.gs_status_label)
        
        # Right column for 3D visualization
        right_column = QVBoxLayout()
        
        # 3D plot container
        plot_group = QGroupBox("3D Surface Plot")
        plot_layout = QVBoxLayout()
        
        # Create the matplotlib canvas for 3D visualization
        self.surface_canvas = MatplotlibCanvas(self.grid_search_tab, width=8, height=8)
        
        # Set up the 3D plot
        from mpl_toolkits.mplot3d import Axes3D
        self.surface_ax = self.surface_canvas.fig.add_subplot(111, projection='3d')
        self.surface_ax.set_xlabel('Population Size')
        self.surface_ax.set_ylabel('Mutation Rate')
        self.surface_ax.set_zlabel('Average Fitness')
        self.surface_ax.set_title('Grid Search Results')
        
        # Add colorbar
        self.surface_canvas.fig.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.viridis), ax=self.surface_ax, 
                                         label='Average Fitness')
        
        plot_layout.addWidget(self.surface_canvas)
        
        # Add plot control buttons
        plot_buttons_layout = QHBoxLayout()
        
        self.update_plot_button = QPushButton("Update 3D Plot")
        self.update_plot_button.clicked.connect(self.update_surface_plot)
        self.update_plot_button.setEnabled(False)
        plot_buttons_layout.addWidget(self.update_plot_button)
        
        self.save_plot_button = QPushButton("Save Plot")
        self.save_plot_button.clicked.connect(self.save_surface_plot)
        self.save_plot_button.setEnabled(False)
        plot_buttons_layout.addWidget(self.save_plot_button)
        
        plot_layout.addLayout(plot_buttons_layout)
        
        plot_group.setLayout(plot_layout)
        right_column.addWidget(plot_group)
        
        # Add columns to the rest layout
        rest_layout.addLayout(left_column, 1)  # 1:1 ratio
        rest_layout.addLayout(right_column, 1)
        
        # Add the rest layout to the main layout
        layout.addLayout(rest_layout)
        
        # Update the grid search info
        self.update_grid_search_info()
    
    def update_grid_search_info(self):
        """Update the grid search information labels with current parameters."""
        # Calculate the number of population size values
        pop_min = self.pop_size_min.value()
        pop_max = self.pop_size_max.value()
        pop_step = self.pop_size_step.value()
        
        if pop_min > pop_max:
            self.pop_size_min.setValue(pop_max)
            pop_min = pop_max
        
        pop_values = range(pop_min, pop_max + 1, pop_step)
        num_pop_values = len(list(pop_values))
        
        # Calculate the number of mutation rate values
        mut_min = self.mut_rate_min.value()
        mut_max = self.mut_rate_max.value()
        mut_step = self.mut_rate_step.value()
        
        if mut_min > mut_max:
            self.mut_rate_min.setValue(mut_max)
            mut_min = mut_max
        
        mut_values = [round(mut_min + i * mut_step, 4) for i in range(int((mut_max - mut_min) / mut_step) + 1) if mut_min + i * mut_step <= mut_max]
        num_mut_values = len(mut_values)
        
        # Calculate the number of elitism values
        elit_min = self.elitism_min.value()
        elit_max = self.elitism_max.value()
        elit_step = self.elitism_step.value()
        
        if elit_min > elit_max:
            self.elitism_min.setValue(elit_max)
            elit_min = elit_max
        
        elit_values = range(elit_min, elit_max + 1, elit_step)
        num_elit_values = len(list(elit_values))
        
        # Calculate the total number of runs
        trials = self.gs_trials.value()
        total_runs = num_pop_values * num_mut_values * num_elit_values * trials
        
        # Update the labels
        self.gs_runs_label.setText(f"Estimated runs: {total_runs}")
        
        # Estimate time (assuming 2 seconds per generation)
        generations = self.gs_generations.value()
        est_time_seconds = total_runs * generations * 2  # rough estimate
        est_time_minutes = est_time_seconds / 60
        
        self.gs_time_label.setText(f"Estimated time: {est_time_minutes:.1f} minutes")
    
    def start_grid_search(self):
        """Start the grid search process for hyperparameter tuning."""
        # Disable the start button and enable the stop button
        self.start_gs_button.setEnabled(False)
        self.stop_gs_button.setEnabled(True)
        self.use_best_button.setEnabled(False)
        self.update_plot_button.setEnabled(False)
        self.save_plot_button.setEnabled(False)
        
        # Clear the results table
        self.gs_results_table.setRowCount(0)
        
        # Calculate the parameter combinations
        pop_min = self.pop_size_min.value()
        pop_max = self.pop_size_max.value()
        pop_step = self.pop_size_step.value()
        pop_values = list(range(pop_min, pop_max + 1, pop_step))
        
        mut_min = self.mut_rate_min.value()
        mut_max = self.mut_rate_max.value()
        mut_step = self.mut_rate_step.value()
        mut_values = [round(mut_min + i * mut_step, 4) for i in range(int((mut_max - mut_min) / mut_step) + 1) if mut_min + i * mut_step <= mut_max]
        
        elit_min = self.elitism_min.value()
        elit_max = self.elitism_max.value()
        elit_step = self.elitism_step.value()
        elit_values = list(range(elit_min, elit_max + 1, elit_step))
        
        trials = self.gs_trials.value()
        generations = self.gs_generations.value()
        
        # Create a list of all parameter combinations
        all_combinations = []
        for pop_size in pop_values:
            for mut_rate in mut_values:
                for elit_count in elit_values:
                    for _ in range(trials):
                        all_combinations.append((pop_size, mut_rate, elit_count))
        
        # Limit to 100 combinations if there are more
        if len(all_combinations) > 100:
            import random
            random.shuffle(all_combinations)  # Shuffle to get a diverse set
            self.gs_param_combinations = all_combinations[:100]
            self.gs_status_label.setText(f"Limited to 100 parameter combinations (from {len(all_combinations)} total)")
        else:
            self.gs_param_combinations = all_combinations
        
        # Initialize the grid search thread
        self.gs_thread = GridSearchThread(
            self.activities,
            self.rooms,
            self.facilitators,
            self.time_slots,
            self.gs_param_combinations,
            generations
        )
        
        # Connect signals
        self.gs_thread.update_progress.connect(self.update_gs_progress)
        self.gs_thread.update_result.connect(self.update_gs_result)
        self.gs_thread.finished_search.connect(self.grid_search_finished)
        
        # Set up the progress bar
        self.gs_progress_bar.setRange(0, len(self.gs_param_combinations))
        self.gs_progress_bar.setValue(0)
        
        # Start the grid search
        self.gs_status_label.setText("Grid search started...")
        self.gs_thread.start()
    
    def update_gs_progress(self, current_run, total_runs, param_str):
        """Update the grid search progress."""
        self.gs_progress_bar.setValue(current_run)
        self.gs_status_label.setText(f"Running combination {current_run}/{total_runs}: {param_str}")
    
    def update_gs_result(self, param_str, avg_fitness, best_fitness, elitism):
        """Add a result to the grid search results table and update the 3D plot."""
        row_position = self.gs_results_table.rowCount()
        self.gs_results_table.insertRow(row_position)
        
        # Add the data to the row
        self.gs_results_table.setItem(row_position, 0, QTableWidgetItem(param_str))
        self.gs_results_table.setItem(row_position, 1, QTableWidgetItem(f"{avg_fitness:.4f}"))
        self.gs_results_table.setItem(row_position, 2, QTableWidgetItem(f"{best_fitness:.4f}"))
        self.gs_results_table.setItem(row_position, 3, QTableWidgetItem(f"{elitism}"))
        
        # Sort the table by average fitness (descending)
        self.gs_results_table.sortItems(1, Qt.DescendingOrder)
        
        # Update the 3D plot periodically instead of after every result to avoid overwhelming the UI
        # Only update every 5 results or when we have very few results
        if row_position < 10 or row_position % 5 == 0:
            self.update_surface_plot()
    
    def stop_grid_search(self):
        """Stop the grid search process."""
        if hasattr(self, 'gs_thread') and self.gs_thread.isRunning():
            self.gs_thread.stop()
            self.gs_status_label.setText("Grid search stopped by user")
            self.start_gs_button.setEnabled(True)
            self.stop_gs_button.setEnabled(False)
            self.use_best_button.setEnabled(True)
            
            # Enable plot buttons if there are results
            if self.gs_results_table.rowCount() > 0:
                self.update_plot_button.setEnabled(True)
                self.save_plot_button.setEnabled(True)
    
    def grid_search_finished(self):
        """Handle the completion of the grid search."""
        self.start_gs_button.setEnabled(True)
        self.stop_gs_button.setEnabled(False)
        self.use_best_button.setEnabled(True)
        self.update_plot_button.setEnabled(True)
        self.save_plot_button.setEnabled(True)
        self.gs_status_label.setText("Grid search completed")
        
        # Update the 3D surface plot
        self.update_surface_plot()
        
        # Show a message
        QMessageBox.information(
            self,
            "Grid Search Complete",
            "Grid search completed. The 3D surface plot has been updated with the results."
        )
    
    def update_surface_plot(self):
        """Update the 3D surface plot with grid search results."""
        if self.gs_results_table.rowCount() == 0:
            return
        
        try:
            # Clear current figure completely to prevent duplicate colorbars
            self.surface_canvas.fig.clear()
            
            # Create a new 3D axes
            self.surface_ax = self.surface_canvas.fig.add_subplot(111, projection='3d')
            
            # Extract data from the results table
            pop_sizes = []
            mut_rates = []
            avg_fitnesses = []
            
            for row in range(self.gs_results_table.rowCount()):
                try:
                    # Parse the parameters
                    param_str = self.gs_results_table.item(row, 0).text()
                    parts = param_str.split(",")
                    
                    pop_size = int(parts[0].split(":")[1].strip())
                    mut_rate = float(parts[1].split(":")[1].strip())
                    
                    avg_fitness = float(self.gs_results_table.item(row, 1).text())
                    
                    pop_sizes.append(pop_size)
                    mut_rates.append(mut_rate)
                    avg_fitnesses.append(avg_fitness)
                except (ValueError, IndexError, AttributeError) as e:
                    print(f"Error parsing row {row}: {str(e)}")
                    continue
            
            if not pop_sizes:  # If no valid data was parsed
                return
            
            # Create 3D scatter plot
            scatter = self.surface_ax.scatter(
                pop_sizes, 
                mut_rates, 
                avg_fitnesses, 
                c=avg_fitnesses, 
                cmap='viridis', 
                s=50, 
                alpha=0.8
            )
            
            # Set labels and title
            self.surface_ax.set_xlabel('Population Size')
            self.surface_ax.set_ylabel('Mutation Rate')
            self.surface_ax.set_zlabel('Average Fitness')
            self.surface_ax.set_title('Grid Search Results')
            
            # Always attempt to create a surface if we have enough points
            if len(pop_sizes) >= 4:
                try:
                    from scipy.interpolate import griddata
                    
                    # Create a grid across the entire parameter space
                    grid_x, grid_y = np.meshgrid(
                        np.linspace(min(pop_sizes), max(pop_sizes), 20),
                        np.linspace(min(mut_rates), max(mut_rates), 20)
                    )
                    
                    # Use all data points for interpolation
                    grid_z = griddata(
                        (pop_sizes, mut_rates), 
                        avg_fitnesses, 
                        (grid_x, grid_y), 
                        method='linear',  # Use linear for more robust interpolation
                        fill_value=np.min(avg_fitnesses)
                    )
                    
                    # Plot the surface
                    surface = self.surface_ax.plot_surface(
                        grid_x, grid_y, grid_z, 
                        cmap='viridis',
                        alpha=0.6,  # More transparent
                        linewidth=0,
                        antialiased=True
                    )
                except Exception as e:
                    print(f"Error creating surface: {str(e)}")
                    # Continue with just the scatter plot
            
            # Add a single colorbar for fitness
            self.surface_canvas.fig.colorbar(scatter, ax=self.surface_ax, label='Fitness')
            
            # Adjust the view to show the data better
            self.surface_ax.view_init(elev=30, azim=45)
            
            # Tight layout to make everything fit
            self.surface_canvas.fig.tight_layout()
            
            # Redraw the canvas
            self.surface_canvas.draw()
            
        except Exception as e:
            import traceback
            print(f"Error creating 3D plot: {str(e)}")
            traceback.print_exc()
            # Don't show an error message every time to avoid dialog spam
            if not hasattr(self, '_plot_error_shown'):
                QMessageBox.warning(
                    self,
                    "Plot Error",
                    f"Error creating 3D plot: {str(e)}"
                )
                self._plot_error_shown = True
    
    def save_surface_plot(self):
        """Save the 3D surface plot to a file."""
        if not hasattr(self, 'surface_canvas'):
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "Save 3D Plot", 
            "grid_search_results.png", 
            "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)"
        )
        
        if file_path:
            try:
                self.surface_canvas.fig.savefig(file_path, dpi=300, bbox_inches='tight')
                QMessageBox.information(
                    self,
                    "Plot Saved",
                    f"3D plot saved to {file_path}"
                )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to save plot: {str(e)}"
                )
    
    def use_best_parameters(self):
        """Apply the best parameters from the grid search to the main controls."""
        if self.gs_results_table.rowCount() > 0:
            # Get the parameters from the first row (best result)
            param_str = self.gs_results_table.item(0, 0).text()
            
            # Parse the parameters
            parts = param_str.split(",")
            pop_size_str = parts[0].strip()
            mut_rate_str = parts[1].strip()
            elit_count_str = parts[2].strip()
            
            pop_size = int(pop_size_str.split(":")[1].strip())
            mut_rate = float(mut_rate_str.split(":")[1].strip())
            elit_count = int(elit_count_str.split(":")[1].strip())
            
            # Apply the parameters to the main controls
            self.population_size_spin.setValue(pop_size)
            self.mutation_rate_spin.setValue(mut_rate)
            
            # Store elitism count for later use
            self.best_elitism_count = elit_count
            
            # Show confirmation
            QMessageBox.information(
                self,
                "Parameters Applied",
                f"Best parameters applied:\nPopulation Size: {pop_size}\nMutation Rate: {mut_rate}\nElitism Count: {elit_count}"
            )
            
            # Switch to the main tab
            self.tabs.setCurrentIndex(0)
        else:
            QMessageBox.warning(
                self,
                "No Results",
                "No grid search results available. Run a grid search first."
            )
    
    def update_schedule_grid(self, schedule):
        """Update the schedule grid with the given schedule."""
        # Clear existing items
        for row in range(self.schedule_grid.rowCount()):
            for col in range(1, self.schedule_grid.columnCount()):
                self.schedule_grid.setItem(row, col, QTableWidgetItem(""))
        
        # Reset facilitator counts
        for i in range(self.facilitator_table.rowCount()):
            self.facilitator_table.setItem(i, 1, QTableWidgetItem("0"))
        
        if not schedule:
            return
        
        # Update with new schedule
        facilitator_counts = {}
        for assignment in schedule.assignments:
            # Find the row (time slot) and column (room)
            time_index = next(i for i, ts in enumerate(self.time_slots) 
                              if ts.hour == assignment.time_slot.hour and ts.minute == assignment.time_slot.minute)
            room_index = next(i for i, r in enumerate(self.rooms) 
                             if r.name == assignment.room.name and r.building == assignment.room.building) + 1  # +1 because column 0 is time slots
            
            # Create an item for the cell
            item = QTableWidgetItem(f"{assignment.activity.name} {assignment.activity.section or ''}\n{assignment.facilitator.name}")
            
            # Color the cell based on room size appropriateness
            if assignment.room.capacity < assignment.activity.expected_enrollment:
                # Room too small - red
                item.setBackground(QColor(255, 200, 200))
            elif assignment.room.capacity > 6 * assignment.activity.expected_enrollment:
                # Room way too big - yellow
                item.setBackground(QColor(255, 255, 200))
            elif assignment.room.capacity > 3 * assignment.activity.expected_enrollment:
                # Room too big - light yellow
                item.setBackground(QColor(255, 255, 230))
            else:
                # Good fit - light green
                item.setBackground(QColor(230, 255, 230))
            
            # Add the item to the grid
            self.schedule_grid.setItem(time_index, room_index, item)
            
            # Update facilitator counts
            facilitator_name = assignment.facilitator.name
            if facilitator_name not in facilitator_counts:
                facilitator_counts[facilitator_name] = 0
            facilitator_counts[facilitator_name] += 1
        
        # Update facilitator table
        for i, facilitator in enumerate(self.facilitators):
            count = facilitator_counts.get(facilitator.name, 0)
            self.facilitator_table.setItem(i, 1, QTableWidgetItem(str(count)))
            
            # Color code based on workload
            item = self.facilitator_table.item(i, 1)
            if count > 4:
                item.setBackground(QColor(255, 200, 200))  # Too many - red
            elif count <= 2 and not facilitator.is_dr_tyler:
                item.setBackground(QColor(255, 200, 200))  # Too few - red
            else:
                item.setBackground(QColor(230, 255, 230))  # Good - green
    
    def update_statistics(self, best_fitness_history, avg_fitness_history):
        """Update the statistics visualization."""
        # Clear the canvas
        self.fitness_canvas.axes.clear()
        
        # Plot the fitness history
        x = range(1, len(best_fitness_history) + 1)
        self.fitness_canvas.axes.plot(x, best_fitness_history, label='Best Fitness')
        self.fitness_canvas.axes.plot(x, avg_fitness_history, label='Average Fitness')
        self.fitness_canvas.axes.set_xlabel('Generation')
        self.fitness_canvas.axes.set_ylabel('Fitness')
        self.fitness_canvas.axes.set_title('Fitness Evolution')
        self.fitness_canvas.axes.legend()
        self.fitness_canvas.axes.grid(True)
        
        # Update the canvas
        self.fitness_canvas.draw()
        
        # Update stats summary
        if best_fitness_history:
            best_fitness = best_fitness_history[-1]
            avg_fitness = avg_fitness_history[-1]
            initial_best = best_fitness_history[0]
            initial_avg = avg_fitness_history[0]
            
            improvement_best = ((best_fitness - initial_best) / abs(initial_best)) * 100 if initial_best != 0 else 0
            improvement_avg = ((avg_fitness - initial_avg) / abs(initial_avg)) * 100 if initial_avg != 0 else 0
            
            stats_text = f"""
            Total Generations: {len(best_fitness_history)}
            
            Initial Best Fitness: {initial_best:.4f}
            Final Best Fitness: {best_fitness:.4f}
            Improvement: {improvement_best:.2f}%
            
            Initial Average Fitness: {initial_avg:.4f}
            Final Average Fitness: {avg_fitness:.4f}
            Improvement: {improvement_avg:.2f}%
            """
            
            self.stats_label.setText(stats_text)
    
    def start_evolution(self):
        """Start the genetic algorithm evolution."""
        # Update UI
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.save_button.setEnabled(False)
        
        # Get parameters
        population_size = self.population_size_spin.value()
        mutation_rate = self.mutation_rate_spin.value()
        max_generations = self.max_gen_spin.value()
        
        # Initialize the progress bar
        self.progress_bar.setRange(0, max_generations)
        self.progress_bar.setValue(0)
        self.generation_label.setText(f"Generation: 0 / {max_generations}")
        
        # Create and start the GA thread
        self.ga_thread = GeneticAlgorithmThread(
            self.activities, 
            self.rooms, 
            self.facilitators, 
            self.time_slots,
            population_size,
            mutation_rate,
            max_generations
        )
        
        # Connect signals
        self.ga_thread.update_progress.connect(self.update_progress)
        self.ga_thread.update_best_schedule.connect(self.update_best_schedule)
        self.ga_thread.finished_evolution.connect(self.evolution_finished)
        self.ga_thread.update_status.connect(self.update_status)
        
        # Start the evolution
        self.ga_thread.start()
    
    def stop_evolution(self):
        """Stop the genetic algorithm evolution."""
        if self.ga_thread and self.ga_thread.isRunning():
            self.ga_thread.stop()
            self.ga_thread.wait()
            
            # Update UI
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.save_button.setEnabled(True)
    
    def update_progress(self, generation, best_fitness, avg_fitness):
        """Update the progress UI with current evolution status."""
        self.progress_bar.setValue(generation)
        self.generation_label.setText(f"Generation: {generation} / {self.max_gen_spin.value()}")
        self.fitness_label.setText(f"Best Fitness: {best_fitness:.4f} | Avg Fitness: {avg_fitness:.4f}")
    
    def update_best_schedule(self, schedule):
        """Update the best schedule display."""
        self.current_best_schedule = schedule
        self.update_schedule_grid(schedule)
    
    def evolution_finished(self, best_schedule, best_fitness_history, avg_fitness_history):
        """Handle the completion of the evolution process."""
        # Update UI
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.save_button.setEnabled(True)
        
        # Update the best schedule
        self.current_best_schedule = best_schedule
        self.update_schedule_grid(best_schedule)
        
        # Update statistics
        self.update_statistics(best_fitness_history, avg_fitness_history)
        
        # No need to switch tabs now as schedule is visible in main tab
        
        # Show completion message
        QMessageBox.information(
            self, 
            "Evolution Complete", 
            f"Genetic algorithm completed after {len(best_fitness_history)} generations.\n"
            f"Best fitness: {best_schedule.fitness:.4f}"
        )
    
    def save_schedule(self):
        """Save the best schedule to a file."""
        if not self.current_best_schedule:
            QMessageBox.warning(self, "No Schedule", "There is no schedule to save.")
            return
        
        # Ask for file path
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "Save Schedule", 
            "best_schedule.txt", 
            "Text Files (*.txt)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write(f"Schedule Fitness: {self.current_best_schedule.fitness:.4f}\n\n")
                    
                    # Group assignments by time slot for better readability
                    by_time = {}
                    for assignment in self.current_best_schedule.assignments:
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
                
                QMessageBox.information(self, "Success", f"Schedule saved to {file_path}")
            
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save schedule: {str(e)}")

    def update_status(self, status):
        """Update the status label with the given status message."""
        self.status_label.setText(status)
        print(f"Status: {status}")  # Also print to console for debugging


def main():
    """Main function to run the visualization."""
    app = QApplication(sys.argv)
    window = ScheduleVisualizer()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main() 