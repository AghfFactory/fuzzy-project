"""
SQLite database for storing user profiles and feature data.
"""

import sqlite3
from typing import List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class UserProfile:
    """User profile data structure."""
    id: int
    name: str
    mfcc_mean: float
    spectral_centroid: float
    zero_crossing_rate: float


class Database:
    """Database manager for speaker profiles."""
    
    def __init__(self, db_path: str = "speaker_profiles.db"):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
    
    def initialize(self):
        """Initialize database connection and create tables."""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
        logger.info(f"Database initialized: {self.db_path}")
    
    def _create_tables(self):
        """Create database tables if they don't exist."""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                mfcc_mean REAL NOT NULL,
                spectral_centroid REAL NOT NULL,
                zero_crossing_rate REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()
        logger.info("Database tables created")
    
    def create_user_profile(
        self,
        name: str,
        mfcc_mean: float,
        spectral_centroid: float,
        zero_crossing_rate: float
    ) -> int:
        """Create a new user profile."""
        cursor = self.conn.cursor()
        
        # Check if user already exists
        cursor.execute("SELECT id FROM users WHERE name = ?", (name,))
        existing = cursor.fetchone()
        
        if existing:
            # Update existing user
            cursor.execute("""
                UPDATE users 
                SET mfcc_mean = ?, spectral_centroid = ?, zero_crossing_rate = ?
                WHERE name = ?
            """, (mfcc_mean, spectral_centroid, zero_crossing_rate, name))
            user_id = existing['id']
        else:
            # Insert new user
            cursor.execute("""
                INSERT INTO users (name, mfcc_mean, spectral_centroid, zero_crossing_rate)
                VALUES (?, ?, ?, ?)
            """, (name, mfcc_mean, spectral_centroid, zero_crossing_rate))
            user_id = cursor.lastrowid
        
        self.conn.commit()
        logger.info(f"User profile saved: {name} (ID: {user_id})")
        return user_id
    
    def get_user_profile(self, user_id: int) -> Optional[UserProfile]:
        """Get user profile by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        
        if row:
            return UserProfile(
                id=row['id'],
                name=row['name'],
                mfcc_mean=row['mfcc_mean'],
                spectral_centroid=row['spectral_centroid'],
                zero_crossing_rate=row['zero_crossing_rate']
            )
        return None
    
    def get_user_by_name(self, name: str) -> Optional[UserProfile]:
        """Get user profile by name."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM users WHERE name = ?", (name,))
        row = cursor.fetchone()
        
        if row:
            return UserProfile(
                id=row['id'],
                name=row['name'],
                mfcc_mean=row['mfcc_mean'],
                spectral_centroid=row['spectral_centroid'],
                zero_crossing_rate=row['zero_crossing_rate']
            )
        return None
    
    def get_all_users(self) -> List[UserProfile]:
        """Get all registered users."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM users ORDER BY name")
        rows = cursor.fetchall()
        
        return [
            UserProfile(
                id=row['id'],
                name=row['name'],
                mfcc_mean=row['mfcc_mean'],
                spectral_centroid=row['spectral_centroid'],
                zero_crossing_rate=row['zero_crossing_rate']
            )
            for row in rows
        ]
    
    def delete_user(self, user_id: int) -> bool:
        """Delete a user profile."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
        self.conn.commit()
        return cursor.rowcount > 0
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

