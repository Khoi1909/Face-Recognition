"""
Face Database Management System
Handles face encoding, storage, and retrieval operations
"""

import os
import pickle
import numpy as np
import face_recognition
import cv2
from typing import List, Dict, Tuple, Optional, Any
import logging
from datetime import datetime
import json
import hashlib
from pathlib import Path

logger = logging.getLogger(__name__)

class FaceDatabase:
    """Manages face encodings database with CRUD operations"""
    
    def __init__(self, database_dir: str = "face_database"):
        self.database_dir = Path(database_dir)
        self.database_dir.mkdir(exist_ok=True)
        
        self.encodings_file = self.database_dir / "face_encodings.pkl"
        self.metadata_file = self.database_dir / "face_metadata.json"
        self.images_dir = self.database_dir / "images"
        self.images_dir.mkdir(exist_ok=True)
        
        self.known_face_encodings = []
        self.known_face_names = []
        self.face_metadata = {}
        
        self._load_database()
        logger.info(f"Face database initialized with {len(self.known_face_names)} faces")
    
    def _generate_face_id(self, name: str) -> str:
        """Generate unique face ID"""
        timestamp = datetime.now().isoformat()
        unique_string = f"{name}_{timestamp}"
        return hashlib.md5(unique_string.encode()).hexdigest()[:8]
    
    def _load_database(self):
        """Load existing face database"""
        try:
            if self.encodings_file.exists():
                with open(self.encodings_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data.get('encodings', [])
                    self.known_face_names = data.get('names', [])
                
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    self.face_metadata = json.load(f)
                    
            logger.info(f"Loaded {len(self.known_face_names)} faces from database")
            
        except Exception as e:
            logger.error(f"Error loading database: {e}")
            self.known_face_encodings = []
            self.known_face_names = []
            self.face_metadata = {}
    
    def _save_database(self):
        """Save face database to disk"""
        try:
            # Save encodings
            data = {
                'encodings': self.known_face_encodings,
                'names': self.known_face_names,
                'version': '1.0',
                'created': datetime.now().isoformat()
            }
            
            with open(self.encodings_file, 'wb') as f:
                pickle.dump(data, f)
            
            # Save metadata
            with open(self.metadata_file, 'w') as f:
                json.dump(self.face_metadata, f, indent=2)
                
            logger.info("Database saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving database: {e}")
    
    def add_face_from_image(self, image_path: str, name: str, 
                           description: str = "") -> bool:
        """Add a face to the database from an image file"""
        try:
            # Load image
            image = face_recognition.load_image_file(image_path)
            
            # Find face locations
            face_locations = face_recognition.face_locations(image)
            
            if not face_locations:
                logger.warning(f"No faces found in {image_path}")
                return False
            
            if len(face_locations) > 1:
                logger.warning(f"Multiple faces found in {image_path}, using the first one")
            
            # Get face encoding
            face_encodings = face_recognition.face_encodings(image, face_locations)
            
            if not face_encodings:
                logger.warning(f"Could not encode face in {image_path}")
                return False
            
            face_encoding = face_encodings[0]
            
            # Generate unique ID
            face_id = self._generate_face_id(name)
            
            # Save image copy
            image_filename = f"{face_id}_{name}.jpg"
            image_save_path = self.images_dir / image_filename
            
            # Convert RGB to BGR for OpenCV
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(image_save_path), image_bgr)
            
            # Add to database
            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(name)
            
            # Add metadata
            self.face_metadata[face_id] = {
                'name': name,
                'description': description,
                'image_path': str(image_save_path),
                'added_date': datetime.now().isoformat(),
                'face_location': face_locations[0],
                'encoding_method': 'dlib_face_recognition'
            }
            
            self._save_database()
            logger.info(f"Added face for {name} with ID {face_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding face from image: {e}")
            return False
    
    def add_face_from_frame(self, frame: np.ndarray, name: str, 
                           description: str = "") -> bool:
        """Add a face to the database from a video frame"""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Find face locations
            face_locations = face_recognition.face_locations(rgb_frame)
            
            if not face_locations:
                logger.warning("No faces found in frame")
                return False
            
            if len(face_locations) > 1:
                logger.warning("Multiple faces found in frame, using the largest one")
                # Select the largest face
                areas = [(loc[2] - loc[0]) * (loc[1] - loc[3]) for loc in face_locations]
                largest_idx = np.argmax(areas)
                face_locations = [face_locations[largest_idx]]
            
            # Get face encoding
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            if not face_encodings:
                logger.warning("Could not encode face in frame")
                return False
            
            face_encoding = face_encodings[0]
            
            # Generate unique ID
            face_id = self._generate_face_id(name)
            
            # Save frame
            image_filename = f"{face_id}_{name}.jpg"
            image_save_path = self.images_dir / image_filename
            cv2.imwrite(str(image_save_path), frame)
            
            # Add to database
            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(name)
            
            # Add metadata
            self.face_metadata[face_id] = {
                'name': name,
                'description': description,
                'image_path': str(image_save_path),
                'added_date': datetime.now().isoformat(),
                'face_location': face_locations[0],
                'encoding_method': 'dlib_face_recognition',
                'source': 'video_frame'
            }
            
            self._save_database()
            logger.info(f"Added face for {name} with ID {face_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding face from frame: {e}")
            return False
    
    def remove_face(self, name: str) -> bool:
        """Remove a face from the database"""
        try:
            indices_to_remove = []
            face_ids_to_remove = []
            
            # Find all faces with this name
            for i, face_name in enumerate(self.known_face_names):
                if face_name == name:
                    indices_to_remove.append(i)
            
            # Find face IDs to remove from metadata
            for face_id, metadata in self.face_metadata.items():
                if metadata['name'] == name:
                    face_ids_to_remove.append(face_id)
            
            if not indices_to_remove:
                logger.warning(f"No faces found for name: {name}")
                return False
            
            # Remove from lists (in reverse order to maintain indices)
            for i in reversed(indices_to_remove):
                del self.known_face_encodings[i]
                del self.known_face_names[i]
            
            # Remove from metadata and delete image files
            for face_id in face_ids_to_remove:
                metadata = self.face_metadata[face_id]
                image_path = Path(metadata['image_path'])
                
                if image_path.exists():
                    image_path.unlink()
                
                del self.face_metadata[face_id]
            
            self._save_database()
            logger.info(f"Removed {len(indices_to_remove)} faces for {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing face: {e}")
            return False
    
    def recognize_faces(self, frame: np.ndarray, tolerance: float = 0.6) -> List[Dict]:
        """Recognize faces in a frame"""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Find face locations and encodings
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            recognized_faces = []
            
            for face_encoding, face_location in zip(face_encodings, face_locations):
                # Compare with known faces
                matches = face_recognition.compare_faces(
                    self.known_face_encodings, face_encoding, tolerance=tolerance
                )
                distances = face_recognition.face_distance(
                    self.known_face_encodings, face_encoding
                )
                
                name = "Unknown"
                confidence = 0.0
                face_id = None
                
                if matches:
                    best_match_index = np.argmin(distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = 1 - distances[best_match_index]
                        
                        # Find face ID
                        for fid, metadata in self.face_metadata.items():
                            if metadata['name'] == name:
                                face_id = fid
                                break
                
                recognized_faces.append({
                    'name': name,
                    'confidence': confidence,
                    'location': face_location,
                    'face_id': face_id,
                    'distance': distances[best_match_index] if matches else 1.0
                })
            
            return recognized_faces
            
        except Exception as e:
            logger.error(f"Error recognizing faces: {e}")
            return []
    
    def get_face_count(self) -> int:
        """Get total number of faces in database"""
        return len(self.known_face_names)
    
    def get_unique_names(self) -> List[str]:
        """Get list of unique names in database"""
        return list(set(self.known_face_names))
    
    def get_face_metadata(self, face_id: str) -> Optional[Dict]:
        """Get metadata for a specific face"""
        return self.face_metadata.get(face_id)
    
    def list_all_faces(self) -> List[Dict]:
        """List all faces in database with metadata"""
        faces = []
        for face_id, metadata in self.face_metadata.items():
            faces.append({
                'face_id': face_id,
                **metadata
            })
        return faces
    
    def export_database(self, export_path: str) -> bool:
        """Export database to a different location"""
        try:
            export_dir = Path(export_path)
            export_dir.mkdir(exist_ok=True)
            
            # Copy encodings file
            import shutil
            shutil.copy2(self.encodings_file, export_dir / "face_encodings.pkl")
            shutil.copy2(self.metadata_file, export_dir / "face_metadata.json")
            
            # Copy images directory
            shutil.copytree(self.images_dir, export_dir / "images", dirs_exist_ok=True)
            
            logger.info(f"Database exported to {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting database: {e}")
            return False
    
    def import_database(self, import_path: str) -> bool:
        """Import database from a different location"""
        try:
            import_dir = Path(import_path)
            
            if not import_dir.exists():
                logger.error(f"Import path does not exist: {import_path}")
                return False
            
            # Backup current database
            backup_dir = self.database_dir.parent / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            import shutil
            shutil.copytree(self.database_dir, backup_dir)
            
            # Import new database
            shutil.copy2(import_dir / "face_encodings.pkl", self.encodings_file)
            shutil.copy2(import_dir / "face_metadata.json", self.metadata_file)
            shutil.copytree(import_dir / "images", self.images_dir, dirs_exist_ok=True)
            
            # Reload database
            self._load_database()
            
            logger.info(f"Database imported from {import_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error importing database: {e}")
            return False
    
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        unique_names = self.get_unique_names()
        name_counts = {}
        
        for name in self.known_face_names:
            name_counts[name] = name_counts.get(name, 0) + 1
        
        return {
            'total_faces': len(self.known_face_names),
            'unique_people': len(unique_names),
            'name_distribution': name_counts,
            'database_size_mb': sum(
                f.stat().st_size for f in self.database_dir.rglob('*') if f.is_file()
            ) / (1024 * 1024),
            'creation_date': min(
                metadata['added_date'] for metadata in self.face_metadata.values()
            ) if self.face_metadata else None
        } 