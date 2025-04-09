#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Data Module for Genetic Algorithm Scheduler
This module provides sample data to test the genetic algorithm scheduling system.
"""

from typing import List, Dict, Tuple
from models import Activity, Room, Facilitator, TimeSlot


def create_facilitators() -> List[Facilitator]:
    """
    Create a list of sample facilitators for testing.
    
    Returns:
        List of Facilitator objects
    """
    facilitators = [
        Facilitator(name="Dr. Tyler", is_dr_tyler=True),
        Facilitator(name="Dr. Smith"),
        Facilitator(name="Dr. Johnson"),
        Facilitator(name="Dr. Williams"),
        Facilitator(name="Dr. Brown"),
        Facilitator(name="Dr. Davis"),
        Facilitator(name="Dr. Miller"),
        Facilitator(name="Dr. Wilson"),
        Facilitator(name="Dr. Moore"),
        Facilitator(name="Dr. Taylor")
    ]
    
    return facilitators


def create_rooms() -> List[Room]:
    """
    Create a list of sample rooms with various capacities and buildings.
    
    Returns:
        List of Room objects
    """
    # Create rooms in different buildings with various capacities
    rooms = [
        # Roman Building
        Room(name="101", capacity=25, building="Roman"),
        Room(name="102", capacity=40, building="Roman"),
        Room(name="103", capacity=35, building="Roman"),
        Room(name="201", capacity=50, building="Roman"),
        Room(name="202", capacity=30, building="Roman"),
        
        # Beach Building
        Room(name="305", capacity=45, building="Beach"),
        Room(name="307", capacity=60, building="Beach"),
        Room(name="309", capacity=75, building="Beach"),
        Room(name="311", capacity=25, building="Beach"),
        
        # Central Building
        Room(name="100", capacity=100, building="Central"),
        Room(name="200", capacity=150, building="Central"),
        Room(name="300", capacity=200, building="Central"),
        
        # West Building
        Room(name="W101", capacity=30, building="West"),
        Room(name="W102", capacity=35, building="West"),
        Room(name="W201", capacity=40, building="West"),
        Room(name="W202", capacity=20, building="West")
    ]
    
    return rooms


def create_time_slots() -> List[TimeSlot]:
    """
    Create time slots for MWF, assuming 50-minute slots.
    
    Returns:
        List of TimeSlot objects
    """
    # Create time slots from 8:00 AM to 5:00 PM
    time_slots = [
        TimeSlot(hour=8, minute=0),    # 8:00 AM
        TimeSlot(hour=9, minute=0),    # 9:00 AM
        TimeSlot(hour=10, minute=0),   # 10:00 AM
        TimeSlot(hour=11, minute=0),   # 11:00 AM
        TimeSlot(hour=12, minute=0),   # 12:00 PM
        TimeSlot(hour=13, minute=0),   # 1:00 PM
        TimeSlot(hour=14, minute=0),   # 2:00 PM
        TimeSlot(hour=15, minute=0),   # 3:00 PM
        TimeSlot(hour=16, minute=0),   # 4:00 PM
        TimeSlot(hour=17, minute=0)    # 5:00 PM
    ]
    
    return time_slots


def create_activities(facilitators: List[Facilitator]) -> List[Activity]:
    """
    Create a list of sample activities with facilitator preferences.
    
    Args:
        facilitators: List of available facilitators to assign preferences
        
    Returns:
        List of Activity objects
    """
    # Helper to get facilitators by name
    def get_facilitator(name: str) -> Facilitator:
        return next(f for f in facilitators if f.name == name)
    
    # Create activities with SLA 101 and 191 having sections A and B
    activities = [
        # SLA 101 sections (special case mentioned in assignment)
        Activity(
            name="SLA 101",
            expected_enrollment=40,
            section="A",
            preferred_facilitators=[
                get_facilitator("Dr. Smith"),
                get_facilitator("Dr. Johnson")
            ],
            other_facilitators=[
                get_facilitator("Dr. Davis"),
                get_facilitator("Dr. Wilson")
            ]
        ),
        Activity(
            name="SLA 101",
            expected_enrollment=40,
            section="B",
            preferred_facilitators=[
                get_facilitator("Dr. Smith"),
                get_facilitator("Dr. Brown")
            ],
            other_facilitators=[
                get_facilitator("Dr. Davis"),
                get_facilitator("Dr. Wilson")
            ]
        ),
        
        # SLA 191 sections (special case mentioned in assignment)
        Activity(
            name="SLA 191",
            expected_enrollment=30,
            section="A",
            preferred_facilitators=[
                get_facilitator("Dr. Johnson"),
                get_facilitator("Dr. Williams")
            ],
            other_facilitators=[
                get_facilitator("Dr. Tyler"),
                get_facilitator("Dr. Miller")
            ]
        ),
        Activity(
            name="SLA 191",
            expected_enrollment=30,
            section="B",
            preferred_facilitators=[
                get_facilitator("Dr. Johnson"),
                get_facilitator("Dr. Williams")
            ],
            other_facilitators=[
                get_facilitator("Dr. Tyler"),
                get_facilitator("Dr. Miller")
            ]
        ),
        
        # Other regular activities
        Activity(
            name="SLA 201",
            expected_enrollment=25,
            preferred_facilitators=[
                get_facilitator("Dr. Davis"),
                get_facilitator("Dr. Tyler")
            ],
            other_facilitators=[
                get_facilitator("Dr. Moore"),
                get_facilitator("Dr. Taylor")
            ]
        ),
        Activity(
            name="SLA 280",
            expected_enrollment=30,
            preferred_facilitators=[
                get_facilitator("Dr. Brown"),
                get_facilitator("Dr. Miller")
            ],
            other_facilitators=[
                get_facilitator("Dr. Wilson"),
                get_facilitator("Dr. Taylor")
            ]
        ),
        Activity(
            name="SLA 301",
            expected_enrollment=20,
            preferred_facilitators=[
                get_facilitator("Dr. Williams"),
                get_facilitator("Dr. Taylor")
            ],
            other_facilitators=[
                get_facilitator("Dr. Smith"),
                get_facilitator("Dr. Brown")
            ]
        ),
        Activity(
            name="SLA 310",
            expected_enrollment=25,
            preferred_facilitators=[
                get_facilitator("Dr. Wilson"),
                get_facilitator("Dr. Moore")
            ],
            other_facilitators=[
                get_facilitator("Dr. Miller"),
                get_facilitator("Dr. Johnson")
            ]
        ),
        Activity(
            name="SLA 320",
            expected_enrollment=35,
            preferred_facilitators=[
                get_facilitator("Dr. Tyler"),
                get_facilitator("Dr. Davis")
            ],
            other_facilitators=[
                get_facilitator("Dr. Smith"),
                get_facilitator("Dr. Brown")
            ]
        ),
        Activity(
            name="SLA 350",
            expected_enrollment=50,
            preferred_facilitators=[
                get_facilitator("Dr. Moore"),
                get_facilitator("Dr. Taylor")
            ],
            other_facilitators=[
                get_facilitator("Dr. Williams"),
                get_facilitator("Dr. Wilson")
            ]
        ),
        Activity(
            name="SLA 390",
            expected_enrollment=60,
            preferred_facilitators=[
                get_facilitator("Dr. Miller"),
                get_facilitator("Dr. Brown")
            ],
            other_facilitators=[
                get_facilitator("Dr. Johnson"),
                get_facilitator("Dr. Davis")
            ]
        ),
        Activity(
            name="SLA 410",
            expected_enrollment=20,
            preferred_facilitators=[
                get_facilitator("Dr. Tyler"),
                get_facilitator("Dr. Smith")
            ],
            other_facilitators=[
                get_facilitator("Dr. Taylor"),
                get_facilitator("Dr. Moore")
            ]
        ),
    ]
    
    return activities


def load_test_data() -> Tuple[List[Activity], List[Room], List[Facilitator], List[TimeSlot]]:
    """
    Load all test data for the genetic algorithm.
    
    Returns:
        Tuple containing lists of activities, rooms, facilitators, and time slots
    """
    facilitators = create_facilitators()
    rooms = create_rooms()
    time_slots = create_time_slots()
    activities = create_activities(facilitators)
    
    return activities, rooms, facilitators, time_slots 