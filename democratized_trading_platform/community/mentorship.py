"""
Mentorship System for Democratized Trading Platform
Connects experienced traders with beginners to accelerate learning.
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

from ..education.learning_path import LearningPath, SkillLevel


class MentorLevel(Enum):
    """Mentor experience levels."""
    BEGINNER = "beginner"  # 1-2 years experience
    INTERMEDIATE = "intermediate"  # 3-5 years experience
    ADVANCED = "advanced"  # 5+ years experience
    EXPERT = "expert"  # 10+ years experience


class MentorshipStatus(Enum):
    """Status of mentorship relationships."""
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class SessionType(Enum):
    """Types of mentorship sessions."""
    CODE_REVIEW = "code_review"
    STRATEGY_DISCUSSION = "strategy_discussion"
    RISK_MANAGEMENT = "risk_management"
    PSYCHOLOGY = "psychology"
    PORTFOLIO_REVIEW = "portfolio_review"
    GENERAL_GUIDANCE = "general_guidance"


@dataclass
class Mentor:
    """Mentor profile."""
    id: str
    name: str
    email: str
    level: MentorLevel
    experience_years: int
    specialties: List[str]
    bio: str
    hourly_rate: float
    availability: Dict[str, List[str]]  # day -> time slots
    rating: float = 0.0
    total_sessions: int = 0
    total_hours: float = 0.0
    reviews: List[Dict[str, Any]] = field(default_factory=list)
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class Mentee:
    """Mentee profile."""
    id: str
    name: str
    email: str
    skill_level: SkillLevel
    learning_goals: List[str]
    current_challenges: List[str]
    preferred_mentor_level: MentorLevel
    budget_per_session: float
    availability: Dict[str, List[str]]
    total_sessions: int = 0
    total_hours: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class MentorshipSession:
    """Individual mentorship session."""
    id: str
    mentor_id: str
    mentee_id: str
    session_type: SessionType
    scheduled_time: datetime
    duration_minutes: int
    status: MentorshipStatus
    notes: str = ""
    recording_url: str = ""
    feedback_mentor: str = ""
    feedback_mentee: str = ""
    rating: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class MentorshipRelationship:
    """Ongoing mentorship relationship."""
    id: str
    mentor_id: str
    mentee_id: str
    status: MentorshipStatus
    start_date: datetime
    end_date: Optional[datetime] = None
    goals: List[str] = field(default_factory=list)
    progress_notes: List[Dict[str, Any]] = field(default_factory=list)
    sessions: List[str] = field(default_factory=list)  # session IDs
    created_at: datetime = field(default_factory=datetime.now)


class MentorshipSystem:
    """
    Comprehensive mentorship system for the democratized trading platform.
    """
    
    def __init__(self):
        self.mentors = {}
        self.mentees = {}
        self.sessions = {}
        self.relationships = {}
        self.logger = logging.getLogger(__name__)
        
        # Load existing data
        self._load_data()
    
    def _load_data(self):
        """Load mentorship data from storage."""
        try:
            # Load mentors
            with open("data/mentors.json", "r") as f:
                mentors_data = json.load(f)
                for mentor_id, mentor_data in mentors_data.items():
                    self.mentors[mentor_id] = Mentor(**mentor_data)
            
            # Load mentees
            with open("data/mentees.json", "r") as f:
                mentees_data = json.load(f)
                for mentee_id, mentee_data in mentees_data.items():
                    self.mentees[mentee_id] = Mentee(**mentee_data)
            
            # Load sessions
            with open("data/sessions.json", "r") as f:
                sessions_data = json.load(f)
                for session_id, session_data in sessions_data.items():
                    self.sessions[session_id] = MentorshipSession(**session_data)
            
            # Load relationships
            with open("data/relationships.json", "r") as f:
                relationships_data = json.load(f)
                for rel_id, rel_data in relationships_data.items():
                    self.relationships[rel_id] = MentorshipRelationship(**rel_data)
            
            self.logger.info("Mentorship data loaded successfully")
            
        except FileNotFoundError:
            self.logger.info("No existing mentorship data found, starting fresh")
        except Exception as e:
            self.logger.error(f"Error loading mentorship data: {e}")
    
    def _save_data(self):
        """Save mentorship data to storage."""
        try:
            # Save mentors
            mentors_data = {mid: mentor.__dict__ for mid, mentor in self.mentors.items()}
            with open("data/mentors.json", "w") as f:
                json.dump(mentors_data, f, indent=2, default=str)
            
            # Save mentees
            mentees_data = {mid: mentee.__dict__ for mid, mentee in self.mentees.items()}
            with open("data/mentees.json", "w") as f:
                json.dump(mentees_data, f, indent=2, default=str)
            
            # Save sessions
            sessions_data = {sid: session.__dict__ for sid, session in self.sessions.items()}
            with open("data/sessions.json", "w") as f:
                json.dump(sessions_data, f, indent=2, default=str)
            
            # Save relationships
            relationships_data = {rid: rel.__dict__ for rid, rel in self.relationships.items()}
            with open("data/relationships.json", "w") as f:
                json.dump(relationships_data, f, indent=2, default=str)
            
            self.logger.info("Mentorship data saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving mentorship data: {e}")
    
    def register_mentor(self, mentor_data: Dict[str, Any]) -> str:
        """Register a new mentor."""
        mentor_id = str(uuid.uuid4())
        
        mentor = Mentor(
            id=mentor_id,
            name=mentor_data["name"],
            email=mentor_data["email"],
            level=MentorLevel(mentor_data["level"]),
            experience_years=mentor_data["experience_years"],
            specialties=mentor_data["specialties"],
            bio=mentor_data["bio"],
            hourly_rate=mentor_data["hourly_rate"],
            availability=mentor_data["availability"]
        )
        
        self.mentors[mentor_id] = mentor
        self._save_data()
        
        self.logger.info(f"New mentor registered: {mentor.name} ({mentor_id})")
        return mentor_id
    
    def register_mentee(self, mentee_data: Dict[str, Any]) -> str:
        """Register a new mentee."""
        mentee_id = str(uuid.uuid4())
        
        mentee = Mentee(
            id=mentee_id,
            name=mentee_data["name"],
            email=mentee_data["email"],
            skill_level=SkillLevel(mentee_data["skill_level"]),
            learning_goals=mentee_data["learning_goals"],
            current_challenges=mentee_data["current_challenges"],
            preferred_mentor_level=MentorLevel(mentee_data["preferred_mentor_level"]),
            budget_per_session=mentee_data["budget_per_session"],
            availability=mentee_data["availability"]
        )
        
        self.mentees[mentee_id] = mentee
        self._save_data()
        
        self.logger.info(f"New mentee registered: {mentee.name} ({mentee_id})")
        return mentee_id
    
    def find_mentors(self, criteria: Dict[str, Any]) -> List[Mentor]:
        """Find mentors matching specific criteria."""
        matching_mentors = []
        
        for mentor in self.mentors.values():
            if not mentor.is_active:
                continue
            
            # Check level preference
            if "level" in criteria:
                if mentor.level.value != criteria["level"]:
                    continue
            
            # Check specialties
            if "specialties" in criteria:
                if not any(specialty in mentor.specialties for specialty in criteria["specialties"]):
                    continue
            
            # Check hourly rate
            if "max_hourly_rate" in criteria:
                if mentor.hourly_rate > criteria["max_hourly_rate"]:
                    continue
            
            # Check availability
            if "availability" in criteria:
                if not self._check_availability_overlap(mentor.availability, criteria["availability"]):
                    continue
            
            # Check rating
            if "min_rating" in criteria:
                if mentor.rating < criteria["min_rating"]:
                    continue
            
            matching_mentors.append(mentor)
        
        # Sort by rating and experience
        matching_mentors.sort(key=lambda m: (m.rating, m.experience_years), reverse=True)
        
        return matching_mentors
    
    def _check_availability_overlap(self, mentor_avail: Dict[str, List[str]], mentee_avail: Dict[str, List[str]]) -> bool:
        """Check if mentor and mentee availability overlaps."""
        for day in mentor_avail:
            if day in mentee_avail:
                mentor_times = set(mentor_avail[day])
                mentee_times = set(mentee_avail[day])
                if mentor_times.intersection(mentee_times):
                    return True
        return False
    
    def request_mentorship(self, mentee_id: str, mentor_id: str, goals: List[str]) -> str:
        """Request a mentorship relationship."""
        if mentee_id not in self.mentees or mentor_id not in self.mentors:
            raise ValueError("Invalid mentee or mentor ID")
        
        relationship_id = str(uuid.uuid4())
        
        relationship = MentorshipRelationship(
            id=relationship_id,
            mentor_id=mentor_id,
            mentee_id=mentee_id,
            status=MentorshipStatus.PENDING,
            start_date=datetime.now(),
            goals=goals
        )
        
        self.relationships[relationship_id] = relationship
        self._save_data()
        
        self.logger.info(f"Mentorship requested: {mentee_id} -> {mentor_id}")
        return relationship_id
    
    def accept_mentorship(self, relationship_id: str) -> bool:
        """Accept a mentorship request."""
        if relationship_id not in self.relationships:
            return False
        
        relationship = self.relationships[relationship_id]
        relationship.status = MentorshipStatus.ACTIVE
        
        self._save_data()
        
        self.logger.info(f"Mentorship accepted: {relationship_id}")
        return True
    
    def schedule_session(
        self, 
        relationship_id: str, 
        session_type: SessionType, 
        scheduled_time: datetime, 
        duration_minutes: int = 60
    ) -> str:
        """Schedule a mentorship session."""
        if relationship_id not in self.relationships:
            raise ValueError("Invalid relationship ID")
        
        relationship = self.relationships[relationship_id]
        if relationship.status != MentorshipStatus.ACTIVE:
            raise ValueError("Relationship is not active")
        
        session_id = str(uuid.uuid4())
        
        session = MentorshipSession(
            id=session_id,
            mentor_id=relationship.mentor_id,
            mentee_id=relationship.mentee_id,
            session_type=session_type,
            scheduled_time=scheduled_time,
            duration_minutes=duration_minutes,
            status=MentorshipStatus.PENDING
        )
        
        self.sessions[session_id] = session
        relationship.sessions.append(session_id)
        
        self._save_data()
        
        self.logger.info(f"Session scheduled: {session_id} for {scheduled_time}")
        return session_id
    
    def complete_session(self, session_id: str, notes: str = "", recording_url: str = "") -> bool:
        """Mark a session as completed."""
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        session.status = MentorshipStatus.COMPLETED
        session.notes = notes
        session.recording_url = recording_url
        
        # Update mentor and mentee statistics
        mentor = self.mentors[session.mentor_id]
        mentee = self.mentees[session.mentee_id]
        
        mentor.total_sessions += 1
        mentor.total_hours += session.duration_minutes / 60
        
        mentee.total_sessions += 1
        mentee.total_hours += session.duration_minutes / 60
        
        self._save_data()
        
        self.logger.info(f"Session completed: {session_id}")
        return True
    
    def rate_session(self, session_id: str, rating: float, feedback: str = "") -> bool:
        """Rate a completed session."""
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        if session.status != MentorshipStatus.COMPLETED:
            return False
        
        session.rating = rating
        session.feedback_mentee = feedback
        
        # Update mentor's average rating
        mentor = self.mentors[session.mentor_id]
        mentor.reviews.append({
            "session_id": session_id,
            "rating": rating,
            "feedback": feedback,
            "date": datetime.now().isoformat()
        })
        
        # Recalculate average rating
        if mentor.reviews:
            mentor.rating = sum(r["rating"] for r in mentor.reviews) / len(mentor.reviews)
        
        self._save_data()
        
        self.logger.info(f"Session rated: {session_id} with {rating} stars")
        return True
    
    def get_mentor_profile(self, mentor_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed mentor profile."""
        if mentor_id not in self.mentors:
            return None
        
        mentor = self.mentors[mentor_id]
        
        # Get recent sessions
        recent_sessions = [
            session for session in self.sessions.values()
            if session.mentor_id == mentor_id and session.status == MentorshipStatus.COMPLETED
        ]
        recent_sessions.sort(key=lambda s: s.scheduled_time, reverse=True)
        
        # Get active relationships
        active_relationships = [
            rel for rel in self.relationships.values()
            if rel.mentor_id == mentor_id and rel.status == MentorshipStatus.ACTIVE
        ]
        
        return {
            "id": mentor.id,
            "name": mentor.name,
            "level": mentor.level.value,
            "experience_years": mentor.experience_years,
            "specialties": mentor.specialties,
            "bio": mentor.bio,
            "hourly_rate": mentor.hourly_rate,
            "availability": mentor.availability,
            "rating": mentor.rating,
            "total_sessions": mentor.total_sessions,
            "total_hours": mentor.total_hours,
            "recent_sessions": len(recent_sessions[:5]),
            "active_mentees": len(active_relationships),
            "reviews": mentor.reviews[-5:]  # Last 5 reviews
        }
    
    def get_mentee_progress(self, mentee_id: str) -> Optional[Dict[str, Any]]:
        """Get mentee learning progress."""
        if mentee_id not in self.mentees:
            return None
        
        mentee = self.mentees[mentee_id]
        
        # Get mentorship relationships
        relationships = [
            rel for rel in self.relationships.values()
            if rel.mentee_id == mentee_id
        ]
        
        # Get completed sessions
        completed_sessions = [
            session for session in self.sessions.values()
            if session.mentee_id == mentee_id and session.status == MentorshipStatus.COMPLETED
        ]
        
        # Calculate progress metrics
        total_mentorship_hours = sum(s.duration_minutes for s in completed_sessions) / 60
        avg_session_rating = sum(s.rating for s in completed_sessions) / len(completed_sessions) if completed_sessions else 0
        
        return {
            "id": mentee.id,
            "name": mentee.name,
            "skill_level": mentee.skill_level.value,
            "learning_goals": mentee.learning_goals,
            "current_challenges": mentee.current_challenges,
            "total_sessions": mentee.total_sessions,
            "total_hours": mentee.total_hours,
            "total_mentorship_hours": total_mentorship_hours,
            "avg_session_rating": avg_session_rating,
            "active_relationships": len([r for r in relationships if r.status == MentorshipStatus.ACTIVE]),
            "completed_relationships": len([r for r in relationships if r.status == MentorshipStatus.COMPLETED])
        }
    
    def get_recommended_mentors(self, mentee_id: str) -> List[Dict[str, Any]]:
        """Get personalized mentor recommendations for a mentee."""
        if mentee_id not in self.mentees:
            return []
        
        mentee = self.mentees[mentee_id]
        
        # Build search criteria
        criteria = {
            "level": mentee.preferred_mentor_level.value,
            "max_hourly_rate": mentee.budget_per_session,
            "availability": mentee.availability,
            "min_rating": 4.0  # Recommend only highly-rated mentors
        }
        
        # Add specialties based on learning goals
        if mentee.learning_goals:
            criteria["specialties"] = mentee.learning_goals
        
        matching_mentors = self.find_mentors(criteria)
        
        # Convert to profile format
        recommendations = []
        for mentor in matching_mentors[:10]:  # Top 10 recommendations
            profile = self.get_mentor_profile(mentor.id)
            if profile:
                recommendations.append(profile)
        
        return recommendations
    
    def generate_mentorship_report(self, relationship_id: str) -> Dict[str, Any]:
        """Generate a comprehensive mentorship report."""
        if relationship_id not in self.relationships:
            return {}
        
        relationship = self.relationships[relationship_id]
        mentor = self.mentors[relationship.mentor_id]
        mentee = self.mentees[relationship.mentee_id]
        
        # Get all sessions for this relationship
        sessions = [
            session for session in self.sessions.values()
            if session.mentor_id == relationship.mentor_id and session.mentee_id == relationship.mentee_id
        ]
        
        completed_sessions = [s for s in sessions if s.status == MentorshipStatus.COMPLETED]
        
        # Calculate metrics
        total_hours = sum(s.duration_minutes for s in completed_sessions) / 60
        avg_rating = sum(s.rating for s in completed_sessions) / len(completed_sessions) if completed_sessions else 0
        
        # Session type breakdown
        session_types = {}
        for session in completed_sessions:
            session_type = session.session_type.value
            session_types[session_type] = session_types.get(session_type, 0) + 1
        
        return {
            "relationship_id": relationship_id,
            "mentor_name": mentor.name,
            "mentee_name": mentee.name,
            "start_date": relationship.start_date.isoformat(),
            "status": relationship.status.value,
            "goals": relationship.goals,
            "total_sessions": len(completed_sessions),
            "total_hours": total_hours,
            "avg_rating": avg_rating,
            "session_types": session_types,
            "progress_notes": relationship.progress_notes,
            "recent_sessions": [
                {
                    "date": s.scheduled_time.isoformat(),
                    "type": s.session_type.value,
                    "duration": s.duration_minutes,
                    "rating": s.rating,
                    "notes": s.notes
                }
                for s in completed_sessions[-5:]  # Last 5 sessions
            ]
        }
    
    def add_progress_note(self, relationship_id: str, note: str, category: str = "general") -> bool:
        """Add a progress note to a mentorship relationship."""
        if relationship_id not in self.relationships:
            return False
        
        relationship = self.relationships[relationship_id]
        
        progress_note = {
            "id": str(uuid.uuid4()),
            "date": datetime.now().isoformat(),
            "category": category,
            "note": note
        }
        
        relationship.progress_notes.append(progress_note)
        self._save_data()
        
        self.logger.info(f"Progress note added to relationship: {relationship_id}")
        return True


# Example usage
if __name__ == "__main__":
    # Initialize mentorship system
    mentorship = MentorshipSystem()
    
    # Register a mentor
    mentor_data = {
        "name": "Sarah Johnson",
        "email": "sarah@example.com",
        "level": "advanced",
        "experience_years": 8,
        "specialties": ["momentum_strategies", "risk_management", "python"],
        "bio": "Experienced algorithmic trader with expertise in momentum strategies and risk management.",
        "hourly_rate": 150.0,
        "availability": {
            "monday": ["09:00", "10:00", "14:00", "15:00"],
            "wednesday": ["09:00", "10:00", "14:00", "15:00"],
            "friday": ["09:00", "10:00", "14:00", "15:00"]
        }
    }
    
    mentor_id = mentorship.register_mentor(mentor_data)
    
    # Register a mentee
    mentee_data = {
        "name": "John Smith",
        "email": "john@example.com",
        "skill_level": "beginner",
        "learning_goals": ["python", "momentum_strategies"],
        "current_challenges": ["understanding backtesting", "risk management"],
        "preferred_mentor_level": "advanced",
        "budget_per_session": 200.0,
        "availability": {
            "monday": ["09:00", "10:00", "14:00"],
            "wednesday": ["09:00", "10:00", "14:00"],
            "friday": ["09:00", "10:00", "14:00"]
        }
    }
    
    mentee_id = mentorship.register_mentee(mentee_data)
    
    # Get mentor recommendations
    recommendations = mentorship.get_recommended_mentors(mentee_id)
    print(f"Found {len(recommendations)} mentor recommendations")
    
    # Request mentorship
    goals = ["Learn Python for trading", "Understand momentum strategies", "Implement proper risk management"]
    relationship_id = mentorship.request_mentorship(mentee_id, mentor_id, goals)
    
    # Accept mentorship
    mentorship.accept_mentorship(relationship_id)
    
    # Schedule a session
    session_time = datetime.now() + timedelta(days=7)
    session_id = mentorship.schedule_session(
        relationship_id, 
        SessionType.STRATEGY_DISCUSSION, 
        session_time, 
        60
    )
    
    print(f"Mentorship relationship established: {relationship_id}")
    print(f"Session scheduled: {session_id}") 