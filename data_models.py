from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List
import sqlite3
import pandas as pd

@dataclass
class Equipment:
    id: int
    name: str
    type: str
    model: Optional[str] = None
    location: Optional[str] = None
    install_date: Optional[datetime] = None
    last_maintenance: Optional[datetime] = None

@dataclass
class SensorReading:
    id: int
    equipment_id: int
    timestamp: datetime
    temperature: Optional[float] = None
    pressure: Optional[float] = None
    vibration: Optional[float] = None
    power_consumption: Optional[float] = None

@dataclass
class MaintenanceRecord:
    id: int
    equipment_id: int
    date: datetime
    technician: str
    issue_description: str
    resolution: str
    parts_used: Optional[str] = None

class DatabaseManager:
    """Enhanced database manager with more robust operations"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with proper indexes and constraints"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Equipment table with constraints
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS equipment (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                type TEXT NOT NULL,
                model TEXT,
                location TEXT,
                install_date DATE,
                last_maintenance DATE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Sensor data with indexes for performance
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sensor_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                equipment_id INTEGER NOT NULL,
                timestamp DATETIME NOT NULL,
                temperature REAL,
                pressure REAL,
                vibration REAL,
                power_consumption REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (equipment_id) REFERENCES equipment (id) ON DELETE CASCADE
            )
        ''')
        
        # Create indexes for better query performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sensor_equipment_time ON sensor_data(equipment_id, timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_equipment_type ON equipment(type)')
        
        # Maintenance logs
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS maintenance_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                equipment_id INTEGER NOT NULL,
                date DATE NOT NULL,
                technician TEXT NOT NULL,
                issue_description TEXT NOT NULL,
                resolution TEXT NOT NULL,
                parts_used TEXT,
                cost REAL,
                downtime_hours REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (equipment_id) REFERENCES equipment (id) ON DELETE CASCADE
            )
        ''')
        
        conn.commit()
        conn.close()