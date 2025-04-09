#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Models for Genetic Algorithm Scheduler
This module contains the shared data classes used throughout the application.
"""

from typing import List, Dict, Tuple, Set, Optional, Union, Any
from dataclasses import dataclass, field

# Constants
POPULATION_SIZE = 500
MAX_GENERATIONS = 1000
MUTATION_RATE = 0.01
MIN_GENERATIONS = 100
IMPROVEMENT_THRESHOLD = 0.01  # 1% improvement threshold


@dataclass
class TimeSlot:
    """Represents a 50-minute time slot on MWF."""
    hour: int
    minute: int
    
    def __str__(self) -> str:
        return f"{self.hour}:{self.minute:02d}"
    
    def is_consecutive(self, other: 'TimeSlot') -> bool:
        """Check if this time slot is consecutive with another."""
        # Convert to minutes since midnight for easier comparison
        this_minutes = self.hour * 60 + self.minute
        other_minutes = other.hour * 60 + other.minute
        
        # Check if slots are consecutive (assuming 50-minute slots with 10-minute breaks)
        return abs(this_minutes - other_minutes) == 60
    
    def hours_apart(self, other: 'TimeSlot') -> float:
        """Calculate how many hours apart two time slots are."""
        # Convert to hours (including fractional part for minutes)
        this_hours = self.hour + self.minute / 60.0
        other_hours = other.hour + other.minute / 60.0
        
        return abs(this_hours - other_hours)


@dataclass
class Room:
    """Represents a room with capacity and building information."""
    name: str
    capacity: int
    building: str
    
    def __str__(self) -> str:
        return f"{self.building} {self.name} (cap: {self.capacity})"


@dataclass
class Facilitator:
    """Represents a facilitator (instructor) who can oversee activities."""
    name: str
    # Special case for Dr. Tyler
    is_dr_tyler: bool = False
    
    def __str__(self) -> str:
        return self.name


@dataclass
class Activity:
    """Represents an SLA activity with enrollment and facilitator preferences."""
    name: str
    expected_enrollment: int
    section: Optional[str] = None  # A or B for activities with multiple sections
    preferred_facilitators: List[Facilitator] = field(default_factory=list)
    other_facilitators: List[Facilitator] = field(default_factory=list)
    
    def __str__(self) -> str:
        section_str = f" Section {self.section}" if self.section else ""
        return f"{self.name}{section_str} (exp: {self.expected_enrollment})"


@dataclass
class ActivityAssignment:
    """Represents the assignment of room, time, and facilitator to an activity."""
    activity: Activity
    room: Room
    time_slot: TimeSlot
    facilitator: Facilitator
    
    def __str__(self) -> str:
        return f"{self.activity} -> {self.room}, {self.time_slot}, {self.facilitator}"


@dataclass
class Schedule:
    """Represents a complete schedule with assignments for all activities."""
    assignments: List[ActivityAssignment] = field(default_factory=list)
    fitness: float = 0.0
    
    def __str__(self) -> str:
        return f"Schedule with {len(self.assignments)} assignments, fitness: {self.fitness:.2f}" 