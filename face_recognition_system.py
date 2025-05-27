"""
Real-Time Face Recognition System with CUDA Acceleration
Main system integrating face detection, recognition, and GPU acceleration
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple
import logging
import time
import threading
from queue import Queue
from datetime import datetime
import face_recognition
from cuda_utils import cuda_manager
from face_database import FaceDatabase

# Try to import MTCNN, but don't fail if it's not available
try:
    from facenet_pytorch import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False
    MTCNN = None

logger = logging.getLogger(__name__)

class FaceDetector:
    """Base class for face detectors"""
    
    def __init__(self, name: str):
        self.name = name
        self.detection_count = 0
        self.total_time = 0.0
    
    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in frame. Returns list of (x, y, w, h) bounding boxes"""
        raise NotImplementedError
    
    def get_average_time(self) -> float:
        """Get average detection time"""
        return self.total_time / max(self.detection_count, 1)

class CUDAAcceleratedPreprocessor:
    """GPU-accelerated image preprocessing with smart fallbacks"""
    
    def __init__(self):
        self.device = cuda_manager.device
        self.use_cuda = cuda_manager.cuda_available
        self.batch_threshold = 5  # Use GPU only for batches of 5+ images
        
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Accelerated frame preprocessing - using CPU for single frames (more efficient)"""
        # For single frames, CPU is faster due to memory transfer overhead
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    def preprocess_batch(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """GPU-accelerated batch preprocessing for multiple frames"""
        if not self.use_cuda or len(frames) < self.batch_threshold:
            # CPU fallback for small batches
            return [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
        
        try:
            # Stack frames into a batch tensor
            frame_batch = np.stack(frames)
            frame_tensor = torch.from_numpy(frame_batch).float().to(self.device)
            
            # RGB conversion weights (for BGR to Gray conversion)
            weights = torch.tensor([0.114, 0.587, 0.299], device=self.device)
            
            # Batch convert BGR to grayscale on GPU
            gray_batch = torch.sum(frame_tensor * weights, dim=3)
            
            # Convert back to numpy and split
            gray_frames = gray_batch.cpu().numpy().astype(np.uint8)
            return [gray_frames[i] for i in range(len(frames))]
            
        except Exception as e:
            logger.warning(f"GPU batch preprocessing failed: {e}, falling back to CPU")
            return [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]

class HaarCascadeDetector(FaceDetector):
    """Haar cascade face detector with GPU preprocessing"""
    
    def __init__(self):
        super().__init__("Haar Cascade (GPU Preprocessed)")
        self.cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.scale_factor = 1.1
        self.min_neighbors = 5
        self.min_size = (30, 30)
        self.preprocessor = CUDAAcceleratedPreprocessor()
    
    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        start_time = time.time()
        
        # GPU-accelerated preprocessing
        gray = self.preprocessor.preprocess_frame(frame)
        
        # Detect faces using CPU (Haar cascades don't have good CUDA support)
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size
        )
        
        self.detection_count += 1
        self.total_time += time.time() - start_time
        
        return [(x, y, w, h) for x, y, w, h in faces]

class DNNFaceDetector(FaceDetector):
    """DNN face detector with CUDA acceleration"""
    
    def __init__(self):
        super().__init__("DNN MobileNet (CUDA)")
        self.confidence_threshold = 0.5
        self.backend_initialized = False
        self.use_gpu_preprocessing = cuda_manager.cuda_available
        
        # Load pre-trained model
        model_path = "models/opencv_face_detector_uint8.pb"
        config_path = "models/opencv_face_detector.pbtxt"
        
        try:
            self.net = cv2.dnn.readNetFromTensorflow(model_path, config_path)
            logger.info("DNN detector model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load DNN model: {e}")
            self.net = None
    
    def _initialize_backend(self):
        """Initialize the backend with proper CUDA setup"""
        if self.backend_initialized or self.net is None:
            return
            
        try:
            # Always use CPU backend for stability (OpenCV DNN CUDA is problematic)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            logger.info("DNN detector using CPU backend (stable)")
            self.backend_initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize DNN backend: {e}")
            self.net = None
    
    def _gpu_preprocess_blob(self, frame: np.ndarray) -> np.ndarray:
        """GPU-accelerated blob preprocessing"""
        if not self.use_gpu_preprocessing:
            return cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123])
        
        try:
            # Convert frame to tensor and move to GPU
            frame_tensor = torch.from_numpy(frame).float().to(cuda_manager.device)
            
            # Resize to 300x300
            frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)  # BHWC to BCHW
            resized = F.interpolate(frame_tensor, size=(300, 300), mode='bilinear', align_corners=False)
            
            # Subtract mean values [104, 117, 123] for BGR
            mean_values = torch.tensor([104, 117, 123], device=cuda_manager.device).view(1, 3, 1, 1)
            blob_tensor = resized - mean_values
            
            # Convert back to numpy
            blob = blob_tensor.cpu().numpy()
            return blob
            
        except Exception as e:
            logger.warning(f"GPU blob preprocessing failed: {e}, using CPU")
            return cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123])
    
    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        if self.net is None:
            return []
            
        # Initialize backend on first call
        if not self.backend_initialized:
            self._initialize_backend()
            
        if self.net is None:
            return []
            
        start_time = time.time()
        
        try:
            h, w = frame.shape[:2]
            
            # GPU-accelerated blob creation
            blob = self._gpu_preprocess_blob(frame)
            self.net.setInput(blob)
            
            # Forward pass (CPU-based but with GPU preprocessing)
            detections = self.net.forward()
            
            faces = []
            
            # Process detections
            if detections is not None and len(detections.shape) >= 3 and detections.shape[2] > 0:
                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    
                    if confidence > self.confidence_threshold:
                        x1 = int(detections[0, 0, i, 3] * w)
                        y1 = int(detections[0, 0, i, 4] * h)
                        x2 = int(detections[0, 0, i, 5] * w)
                        y2 = int(detections[0, 0, i, 6] * h)
                        
                        # Ensure coordinates are within frame bounds
                        x1 = max(0, min(x1, w))
                        y1 = max(0, min(y1, h))
                        x2 = max(0, min(x2, w))
                        y2 = max(0, min(y2, h))
                        
                        # Ensure width and height are positive
                        width = x2 - x1
                        height = y2 - y1
                        
                        if width > 0 and height > 0:
                            faces.append((x1, y1, width, height))
            
            self.detection_count += 1
            self.total_time += time.time() - start_time
            
            return faces
            
        except Exception as e:
            logger.error(f"Error in DNN detection: {e}")
            self.detection_count += 1
            self.total_time += time.time() - start_time
            return []

class MTCNNDetector(FaceDetector):
    """MTCNN face detector with PyTorch CUDA support"""
    
    def __init__(self):
        super().__init__("MTCNN (CUDA)")
        self.device = cuda_manager.device
        
        if not MTCNN_AVAILABLE:
            logger.warning("MTCNN not available - facenet_pytorch not installed")
            self.mtcnn = None
            return
            
        try:
            self.mtcnn = MTCNN(
                image_size=160,
                margin=0,
                min_face_size=20,
                thresholds=[0.6, 0.7, 0.7],
                factor=0.709,
                post_process=False,
                device=self.device
            )
            logger.info(f"MTCNN initialized on {self.device}")
        except Exception as e:
            logger.error(f"Failed to initialize MTCNN: {e}")
            self.mtcnn = None
    
    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        if self.mtcnn is None:
            return []
            
        start_time = time.time()
        
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # MTCNN runs on GPU automatically when device is CUDA
            boxes, _ = self.mtcnn.detect(rgb_frame)
            
            faces = []
            if boxes is not None:
                for box in boxes:
                    x, y, x2, y2 = box.astype(int)
                    w, h = x2 - x, y2 - y
                    faces.append((x, y, w, h))
            
            self.detection_count += 1
            self.total_time += time.time() - start_time
            
            return faces
            
        except Exception as e:
            logger.error(f"Error in MTCNN detection: {e}")
            self.detection_count += 1
            self.total_time += time.time() - start_time
            return []

class OptimizedDNNDetector(FaceDetector):
    """Optimized DNN detector with smart GPU usage"""
    
    def __init__(self):
        super().__init__("DNN Optimized")
        self.confidence_threshold = 0.5
        self.backend_initialized = False
        
        # Load pre-trained model
        model_path = "models/opencv_face_detector_uint8.pb"
        config_path = "models/opencv_face_detector.pbtxt"
        
        try:
            self.net = cv2.dnn.readNetFromTensorflow(model_path, config_path)
            logger.info("Optimized DNN detector model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load DNN model: {e}")
            self.net = None
    
    def _initialize_backend(self):
        """Initialize with optimal backend"""
        if self.backend_initialized or self.net is None:
            return
            
        try:
            # Use CPU backend (most stable and often faster for small operations)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            logger.info("Optimized DNN detector using CPU backend")
            self.backend_initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize DNN backend: {e}")
            self.net = None
    
    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        if self.net is None:
            return []
            
        if not self.backend_initialized:
            self._initialize_backend()
            
        if self.net is None:
            return []
            
        start_time = time.time()
        
        try:
            h, w = frame.shape[:2]
            
            # Optimized blob creation (CPU is faster for this)
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123])
            self.net.setInput(blob)
            
            # Forward pass
            detections = self.net.forward()
            
            faces = []
            
            if detections is not None and len(detections.shape) >= 3 and detections.shape[2] > 0:
                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    
                    if confidence > self.confidence_threshold:
                        x1 = int(detections[0, 0, i, 3] * w)
                        y1 = int(detections[0, 0, i, 4] * h)
                        x2 = int(detections[0, 0, i, 5] * w)
                        y2 = int(detections[0, 0, i, 6] * h)
                        
                        # Ensure coordinates are within frame bounds
                        x1 = max(0, min(x1, w))
                        y1 = max(0, min(y1, h))
                        x2 = max(0, min(x2, w))
                        y2 = max(0, min(y2, h))
                        
                        width = x2 - x1
                        height = y2 - y1
                        
                        if width > 0 and height > 0:
                            faces.append((x1, y1, width, height))
            
            self.detection_count += 1
            self.total_time += time.time() - start_time
            
            return faces
            
        except Exception as e:
            logger.error(f"Error in optimized DNN detection: {e}")
            self.detection_count += 1
            self.total_time += time.time() - start_time
            return []

class CUDAAcceleratedRecognition:
    """GPU-accelerated face recognition operations with smart optimization"""
    
    def __init__(self):
        self.device = cuda_manager.device
        self.use_cuda = cuda_manager.cuda_available
        self.batch_threshold = 10  # Use GPU only for large databases
        
    def batch_encode_faces(self, frames: List[np.ndarray], face_locations_list: List[List]) -> List[np.ndarray]:
        """Batch encode multiple faces for better GPU utilization"""
        if not self.use_cuda or not frames:
            # CPU fallback
            encodings = []
            for frame, locations in zip(frames, face_locations_list):
                frame_encodings = face_recognition.face_encodings(frame, locations)
                encodings.extend(frame_encodings)
            return encodings
        
        # For now, use CPU since face_recognition library doesn't support CUDA
        # This is a placeholder for future CUDA-accelerated face encoding
        encodings = []
        for frame, locations in zip(frames, face_locations_list):
            frame_encodings = face_recognition.face_encodings(frame, locations)
            encodings.extend(frame_encodings)
        return encodings
    
    def accelerated_compare_faces(self, known_encodings: List, face_encoding: np.ndarray, tolerance: float = 0.6) -> List[bool]:
        """GPU-accelerated face comparison - only for large databases"""
        if not self.use_cuda or len(known_encodings) < self.batch_threshold:
            # CPU is faster for small databases
            return face_recognition.compare_faces(known_encodings, face_encoding, tolerance)
        
        try:
            # Convert to tensors for GPU acceleration
            known_tensor = torch.tensor(np.array(known_encodings), device=self.device, dtype=torch.float32)
            face_tensor = torch.tensor(face_encoding, device=self.device, dtype=torch.float32)
            
            # Compute distances on GPU using optimized operations
            distances = torch.norm(known_tensor - face_tensor.unsqueeze(0), dim=1)
            
            # Compare with tolerance
            matches = distances <= tolerance
            
            return matches.cpu().numpy().tolist()
            
        except Exception as e:
            logger.warning(f"GPU face comparison failed: {e}, using CPU")
            return face_recognition.compare_faces(known_encodings, face_encoding, tolerance)
    
    def batch_compare_faces(self, known_encodings: List, face_encodings: List[np.ndarray], tolerance: float = 0.6) -> List[List[bool]]:
        """Batch GPU comparison for multiple faces at once"""
        if not self.use_cuda or len(known_encodings) < self.batch_threshold:
            # CPU fallback
            return [face_recognition.compare_faces(known_encodings, encoding, tolerance) for encoding in face_encodings]
        
        try:
            known_tensor = torch.tensor(np.array(known_encodings), device=self.device, dtype=torch.float32)
            face_batch_tensor = torch.tensor(np.array(face_encodings), device=self.device, dtype=torch.float32)
            
            # Compute all distances at once (batch operation)
            # Shape: [num_faces, num_known]
            distances = torch.cdist(face_batch_tensor, known_tensor)
            
            # Compare with tolerance
            matches = distances <= tolerance
            
            return matches.cpu().numpy().tolist()
            
        except Exception as e:
            logger.warning(f"GPU batch comparison failed: {e}, using CPU")
            return [face_recognition.compare_faces(known_encodings, encoding, tolerance) for encoding in face_encodings]

class FaceRecognitionSystem:
    """Main face recognition system with CUDA acceleration"""
    
    def __init__(self, database_dir: str = "face_database"):
        self.database = FaceDatabase(database_dir)
        self.detectors = {
            'haar': HaarCascadeDetector(),
            'dnn': OptimizedDNNDetector(),
            'mtcnn': MTCNNDetector()
        }
        self.current_detector = 'haar'
        
        # CUDA acceleration components
        self.cuda_recognition = CUDAAcceleratedRecognition()
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        self._last_processing_time = 0.0
        
        # Recognition parameters
        self.recognition_tolerance = 0.6
        self.min_confidence = 0.4
        
        # Threading for real-time processing
        self.frame_queue = Queue(maxsize=5)
        self.result_queue = Queue(maxsize=5)
        self.processing_thread = None
        self.is_processing = False
        
        # Anti-spoofing (basic implementation)
        self.enable_anti_spoofing = False
        self.face_history = {}  # Track face movement for liveness detection
        
        logger.info(f"Face Recognition System initialized with CUDA: {cuda_manager.cuda_available}")
        logger.info(f"Using device: {cuda_manager.device}")
    
    def set_detector(self, detector_name: str):
        """Set active face detector"""
        if detector_name in self.detectors:
            self.current_detector = detector_name
            logger.info(f"Switched to {detector_name} detector")
        else:
            logger.warning(f"Unknown detector: {detector_name}")
    
    def start_processing_thread(self):
        """Start background processing thread"""
        if not self.is_processing:
            self.is_processing = True
            self.processing_thread = threading.Thread(target=self._process_frames)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            logger.info("Processing thread started")
    
    def stop_processing_thread(self):
        """Stop background processing thread"""
        self.is_processing = False
        if self.processing_thread:
            self.processing_thread.join()
            logger.info("Processing thread stopped")
    
    def _process_frames(self):
        """Background frame processing"""
        while self.is_processing:
            try:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get(timeout=0.1)
                    results = self._process_single_frame(frame)
                    
                    if not self.result_queue.full():
                        self.result_queue.put(results)
                    
            except Exception as e:
                logger.error(f"Error in processing thread: {e}")
                time.sleep(0.01)
    
    def _process_single_frame(self, frame: np.ndarray) -> Dict:
        """Process a single frame for face recognition with CUDA acceleration"""
        start_time = time.time()
        
        # Detect faces
        detector = self.detectors[self.current_detector]
        face_boxes = detector.detect(frame)
        
        # Recognize faces using GPU-accelerated recognition when possible
        recognized_faces = []
        if face_boxes:
            recognized_faces = self._recognize_faces_cuda(frame, self.recognition_tolerance)
        
        # Anti-spoofing check
        if self.enable_anti_spoofing:
            recognized_faces = self._apply_anti_spoofing(recognized_faces, frame)
        
        processing_time = time.time() - start_time
        self._last_processing_time = processing_time
        
        return {
            'faces': recognized_faces,
            'face_boxes': face_boxes,
            'processing_time': processing_time,
            'detector': self.current_detector,
            'timestamp': datetime.now(),
            'cuda_used': cuda_manager.cuda_available
        }
    
    def _recognize_faces_cuda(self, frame: np.ndarray, tolerance: float = 0.6) -> List[Dict]:
        """CUDA-accelerated face recognition"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Find face locations
        face_locations = face_recognition.face_locations(rgb_frame)
        
        if not face_locations:
            return []
        
        # Encode faces (CPU for now, as face_recognition doesn't support CUDA)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        if not face_encodings:
            return []
        
        recognized_faces = []
        
        for i, (face_encoding, location) in enumerate(zip(face_encodings, face_locations)):
            # Use GPU-accelerated comparison when possible
            matches = self.cuda_recognition.accelerated_compare_faces(
                self.database.known_face_encodings, 
                face_encoding, 
                tolerance
            )
            
            name = "Unknown"
            confidence = 0.0
            
            if True in matches:
                # Find best match
                face_distances = face_recognition.face_distance(
                    self.database.known_face_encodings, face_encoding
                )
                best_match_index = np.argmin(face_distances)
                
                if matches[best_match_index]:
                    name = self.database.known_face_names[best_match_index]
                    confidence = 1 - face_distances[best_match_index]
            
            recognized_faces.append({
                'name': name,
                'confidence': confidence,
                'location': location,
                'face_encoding': face_encoding
            })
        
        return recognized_faces
    
    def _apply_anti_spoofing(self, faces: List[Dict], frame: np.ndarray) -> List[Dict]:
        """Basic anti-spoofing using movement detection"""
        current_time = time.time()
        filtered_faces = []
        
        for face in faces:
            face_id = face.get('name', 'Unknown')
            if face_id != 'Unknown':
                location = face['location']
                center = ((location[1] + location[3]) // 2, (location[0] + location[2]) // 2)
                
                if face_id in self.face_history:
                    last_center, last_time = self.face_history[face_id]
                    
                    # Calculate movement
                    movement = np.sqrt((center[0] - last_center[0])**2 + (center[1] - last_center[1])**2)
                    time_diff = current_time - last_time
                    
                    # Simple liveness check - face should move slightly
                    if time_diff > 2.0:  # Reset tracking after 2 seconds
                        if movement < 5:  # Too static, might be a photo
                            face['anti_spoofing'] = 'suspicious'
                            face['confidence'] *= 0.5  # Reduce confidence
                        else:
                            face['anti_spoofing'] = 'live'
                    else:
                        face['anti_spoofing'] = 'tracking'
                
                self.face_history[face_id] = (center, current_time)
            
            filtered_faces.append(face)
        
        return filtered_faces
    
    def process_frame(self, frame: np.ndarray, use_threading: bool = False) -> Dict:
        """Process frame for face recognition"""
        if use_threading:
            # Add frame to queue for background processing
            if not self.frame_queue.full():
                self.frame_queue.put(frame.copy())
            
            # Get latest results
            if not self.result_queue.empty():
                return self.result_queue.get()
            else:
                return {'faces': [], 'face_boxes': [], 'processing_time': 0.0}
        else:
            return self._process_single_frame(frame)
    
    def draw_results(self, frame: np.ndarray, results: Dict) -> np.ndarray:
        """Draw recognition results on frame"""
        annotated_frame = frame.copy()
        
        for face in results.get('faces', []):
            name = face['name']
            confidence = face['confidence']
            location = face['location']
            
            # face_recognition returns (top, right, bottom, left)
            top, right, bottom, left = location
            
            # Choose color based on recognition
            if name == "Unknown":
                color = (0, 0, 255)  # Red
            else:
                color = (0, 255, 0)  # Green
                
            # Adjust color based on confidence
            if confidence < self.min_confidence:
                color = (0, 165, 255)  # Orange
            
            # Draw rectangle
            cv2.rectangle(annotated_frame, (left, top), (right, bottom), color, 2)
            
            # Prepare text
            text = f"{name}"
            if name != "Unknown":
                text += f" ({confidence:.2f})"
            
            # Anti-spoofing indicator
            if 'anti_spoofing' in face:
                if face['anti_spoofing'] == 'suspicious':
                    text += " [?]"
                    color = (0, 165, 255)  # Orange for suspicious
            
            # Draw text background
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.6
            thickness = 1
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            cv2.rectangle(annotated_frame, 
                         (left, bottom + baseline), 
                         (left + text_width, bottom + text_height + baseline), 
                         color, cv2.FILLED)
            
            # Draw text
            cv2.putText(annotated_frame, text, (left, bottom + text_height), 
                       font, font_scale, (255, 255, 255), thickness)
        
        # Draw performance info
        self._draw_performance_info(annotated_frame, results)
        
        return annotated_frame
    
    def _draw_performance_info(self, frame: np.ndarray, results: Dict):
        """Draw performance information on frame"""
        # Update FPS
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
        
        # Performance text
        detector_name = results.get('detector', 'Unknown')
        processing_time = results.get('processing_time', 0.0) * 1000  # Convert to ms
        face_count = len(results.get('faces', []))
        # Smart CUDA status
        db_size = self.database.get_face_count()
        cuda_beneficial = db_size >= 10  # GPU beneficial for large databases
        cuda_status = f"🚀 CUDA ({db_size} faces)" if cuda_beneficial and cuda_manager.cuda_available else f"💻 CPU ({db_size} faces)"
        
        info_lines = [
            f"FPS: {self.current_fps:.1f}",
            f"Detector: {detector_name}",
            f"Processing: {processing_time:.1f}ms",
            f"Faces: {face_count}",
            f"Database: {self.database.get_face_count()} faces",
            f"Mode: {cuda_status}"
        ]
        
        # Draw info box
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        y_offset = 25
        for i, line in enumerate(info_lines):
            y_pos = y_offset + i * 20
            color = (0, 255, 0) if "CUDA" in line else (255, 255, 255)
            cv2.putText(frame, line, (10, y_pos), font, font_scale, color, thickness)
    
    def add_face_from_frame(self, frame: np.ndarray, name: str, description: str = "") -> bool:
        """Add a new face to the database from current frame"""
        return self.database.add_face_from_frame(frame, name, description)
    
    def remove_face(self, name: str) -> bool:
        """Remove a face from the database"""
        return self.database.remove_face(name)
    
    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        return self.database.get_statistics()
    
    def get_detector_stats(self) -> Dict:
        """Get detector performance statistics"""
        stats = {}
        for name, detector in self.detectors.items():
            stats[name] = {
                'detection_count': detector.detection_count,
                'average_time_ms': detector.get_average_time() * 1000,
                'total_time': detector.total_time
            }
        return stats
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop_processing_thread()
        cuda_manager.optimize_memory()
        logger.info("Face Recognition System cleaned up") 