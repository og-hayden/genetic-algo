"""
Fitness Evaluation Module for the Genetic Algorithm Scheduler
This module contains the implementation of the fitness function for evaluating schedules.
"""

from typing import Dict, List, Set, Tuple
from dataclasses import dataclass

# Import the data classes from models
from models import Schedule, ActivityAssignment


class FitnessEvaluator:
    """
    Comprehensive fitness evaluator that calculates fitness scores for schedules
    based on all the criteria specified in the assignment.
    """
    
    def __init__(self):
        # Special activity names that have specific rules
        self.sla_101 = "SLA 101"
        self.sla_191 = "SLA 191"
        
        # Buildings with specific consecutive activity rules
        self.special_buildings = {"Roman", "Beach"}
    
    def evaluate(self, schedule: Schedule) -> float:
        """
        Calculate the fitness score for a complete schedule.
        
        Args:
            schedule: The schedule to evaluate
            
        Returns:
            float: The total fitness score
        """
        # Start with a clean slate
        total_fitness = 0.0
        
        # Process data into more convenient formats for analysis
        assignments_by_room_time = self._group_by_room_and_time(schedule)
        assignments_by_facilitator = self._group_by_facilitator(schedule)
        assignments_by_facilitator_time = self._group_by_facilitator_and_time(schedule)
        sla_101_assignments = self._find_activity_assignments(schedule, self.sla_101)
        sla_191_assignments = self._find_activity_assignments(schedule, self.sla_191)
        
        # Evaluate each activity assignment
        for assignment in schedule.assignments:
            activity_fitness = 0.0
            
            # 1. Room conflict check
            activity_fitness += self._evaluate_room_conflicts(assignment, assignments_by_room_time)
            
            # 2. Room size appropriateness
            activity_fitness += self._evaluate_room_size(assignment)
            
            # 3. Facilitator preference
            activity_fitness += self._evaluate_facilitator_preference(assignment)
            
            # 4. Facilitator load
            activity_fitness += self._evaluate_facilitator_load(
                assignment, 
                assignments_by_facilitator,
                assignments_by_facilitator_time
            )
            
            # Add to total
            total_fitness += activity_fitness
        
        # 5. Activity-specific adjustments for SLA 101 and SLA 191
        total_fitness += self._evaluate_sla_specific_rules(
            sla_101_assignments,
            sla_191_assignments
        )
        
        # Set the fitness on the schedule object and return it
        schedule.fitness = total_fitness
        return total_fitness
    
    def _group_by_room_and_time(self, schedule: Schedule) -> Dict[Tuple[str, str], List[ActivityAssignment]]:
        """Group assignments by room and time for conflict detection."""
        result = {}
        for assignment in schedule.assignments:
            key = (assignment.room.name, str(assignment.time_slot))
            if key not in result:
                result[key] = []
            result[key].append(assignment)
        return result
    
    def _group_by_facilitator(self, schedule: Schedule) -> Dict[str, List[ActivityAssignment]]:
        """Group assignments by facilitator for workload analysis."""
        result = {}
        for assignment in schedule.assignments:
            key = assignment.facilitator.name
            if key not in result:
                result[key] = []
            result[key].append(assignment)
        return result
    
    def _group_by_facilitator_and_time(self, schedule: Schedule) -> Dict[Tuple[str, str], List[ActivityAssignment]]:
        """Group assignments by facilitator and time for concurrent activities."""
        result = {}
        for assignment in schedule.assignments:
            key = (assignment.facilitator.name, str(assignment.time_slot))
            if key not in result:
                result[key] = []
            result[key].append(assignment)
        return result
    
    def _find_activity_assignments(self, schedule: Schedule, activity_name: str) -> List[ActivityAssignment]:
        """Find all assignments for a specific activity."""
        return [a for a in schedule.assignments if a.activity.name == activity_name]
    
    def _evaluate_room_conflicts(self, 
                                assignment: ActivityAssignment, 
                                assignments_by_room_time: Dict[Tuple[str, str], List[ActivityAssignment]]) -> float:
        """
        Check if the activity is scheduled in the same room at the same time as another activity.
        
        Returns:
            float: -0.5 if there's a conflict, 0.0 otherwise
        """
        key = (assignment.room.name, str(assignment.time_slot))
        if key in assignments_by_room_time and len(assignments_by_room_time[key]) > 1:
            return -0.5
        return 0.0
    
    def _evaluate_room_size(self, assignment: ActivityAssignment) -> float:
        """
        Evaluate the appropriateness of room size for the activity.
        
        Returns:
            float: 
                -0.5 if room is too small
                -0.2 if room capacity > 3x enrollment
                -0.4 if room capacity > 6x enrollment
                +0.3 otherwise (good fit)
        """
        enrollment = assignment.activity.expected_enrollment
        capacity = assignment.room.capacity
        
        if capacity < enrollment:
            return -0.5  # Room too small
        elif capacity > 6 * enrollment:
            return -0.4  # Room waaaay too big - what were they thinking?
        elif capacity > 3 * enrollment:
            return -0.2  # Room too big
        else:
            return 0.3   # Good fit
    
    def _evaluate_facilitator_preference(self, assignment: ActivityAssignment) -> float:
        """
        Evaluate if the activity is overseen by a preferred or other listed facilitator.
        
        Returns:
            float:
                +0.5 if using a preferred facilitator
                +0.2 if using another listed facilitator
                -0.1 if using any other facilitator
        """
        preferred_names = [f.name for f in assignment.activity.preferred_facilitators]
        other_names = [f.name for f in assignment.activity.other_facilitators]
        
        if assignment.facilitator.name in preferred_names:
            return 0.5
        elif assignment.facilitator.name in other_names:
            return 0.2
        else:
            return -0.1
    
    def _evaluate_facilitator_load(self, 
                                  assignment: ActivityAssignment,
                                  assignments_by_facilitator: Dict[str, List[ActivityAssignment]],
                                  assignments_by_facilitator_time: Dict[Tuple[str, str], List[ActivityAssignment]]) -> float:
        """
        Evaluate the facilitator's workload.
        
        Returns:
            float: Sum of various adjustments based on facilitator load
        """
        result = 0.0
        facilitator_name = assignment.facilitator.name
        key = (facilitator_name, str(assignment.time_slot))
        
        # Check if facilitator has only 1 activity in this time slot
        if key in assignments_by_facilitator_time and len(assignments_by_facilitator_time[key]) == 1:
            result += 0.2
        
        # Check if facilitator has more than one activity at the same time
        if key in assignments_by_facilitator_time and len(assignments_by_facilitator_time[key]) > 1:
            result -= 0.2
        
        # Check total activities for this facilitator
        total_activities = len(assignments_by_facilitator.get(facilitator_name, []))
        
        # Facilitator is scheduled to oversee more than 4 activities total
        if total_activities > 4:
            result -= 0.5
        
        # Facilitator is scheduled to oversee 1 or 2 activities
        # Exception for Dr. Tyler
        if 1 <= total_activities <= 2:
            if not assignment.facilitator.is_dr_tyler:
                result -= 0.4
        
        return result
    
    def _evaluate_sla_specific_rules(self, 
                                    sla_101_assignments: List[ActivityAssignment],
                                    sla_191_assignments: List[ActivityAssignment]) -> float:
        """
        Evaluate the special rules for SLA 101 and SLA 191.
        
        Returns:
            float: Sum of adjustments based on SLA-specific rules
        """
        result = 0.0
        
        # Check section spacing for SLA 101
        result += self._evaluate_section_spacing(sla_101_assignments)
        
        # Check section spacing for SLA 191
        result += self._evaluate_section_spacing(sla_191_assignments)
        
        # Evaluate relationships between SLA 101 and SLA 191
        result += self._evaluate_sla_101_191_relationship(sla_101_assignments, sla_191_assignments)
        
        return result
    
    def _evaluate_section_spacing(self, activity_assignments: List[ActivityAssignment]) -> float:
        """
        Evaluate the spacing between sections of the same activity.
        
        Returns:
            float:
                +0.5 if sections are more than 4 hours apart
                -0.5 if sections are in the same time slot
                0.0 otherwise
        """
        if len(activity_assignments) != 2:
            return 0.0  # Not exactly 2 sections
        
        assignment1, assignment2 = activity_assignments
        time1, time2 = assignment1.time_slot, assignment2.time_slot
        
        # Check if they're in the same time slot
        if str(time1) == str(time2):
            return -0.5
        
        # Check if they're more than 4 hours apart
        hours_apart = time1.hours_apart(time2)
        if hours_apart > 4:
            return 0.5
        
        return 0.0
    
    def _evaluate_sla_101_191_relationship(self, 
                                          sla_101_assignments: List[ActivityAssignment],
                                          sla_191_assignments: List[ActivityAssignment]) -> float:
        """
        Evaluate the relationship between SLA 101 and SLA 191 assignments.
        
        Returns:
            float: Sum of adjustments based on the relationship
        """
        result = 0.0
        
        for sla_101 in sla_101_assignments:
            for sla_191 in sla_191_assignments:
                time1, time2 = sla_101.time_slot, sla_191.time_slot
                
                # Same time slot
                if str(time1) == str(time2):
                    result -= 0.25
                    continue
                
                # Consecutive time slots
                if time1.is_consecutive(time2):
                    result += 0.5
                    
                    # Check for building constraints in consecutive case
                    building1 = sla_101.room.building
                    building2 = sla_191.room.building
                    
                    if ((building1 in self.special_buildings and building2 not in self.special_buildings) or
                        (building2 in self.special_buildings and building1 not in self.special_buildings)):
                        result -= 0.4
                
                # Separated by 1 hour
                hours_apart = abs(time1.hours_apart(time2))
                if 1.0 <= hours_apart < 2.0:
                    result += 0.25
        
        return result 