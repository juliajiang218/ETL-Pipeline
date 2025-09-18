"""
Streaming Data Handler for Real-time User Interactions
Processes real-time user rating and interaction events.
"""

import logging
import json
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from collections import deque
import threading
import time
import pandas as pd
from queue import Queue, Empty


class StreamingDataHandler:
    """Real-time user interaction data processing with buffering and batch processing."""
    
    def __init__(self, buffer_size: int = 10000, batch_size: int = 100, 
                 flush_interval: int = 30):
        self.logger = logging.getLogger(__name__)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        
        # Data buffers
        self.rating_buffer = deque(maxlen=buffer_size)
        self.interaction_buffer = deque(maxlen=buffer_size)
        self.event_queue = Queue()
        
        # Processing state
        self.is_running = False
        self.last_flush_time = datetime.now()
        self.processing_thread = None
        
        # Callbacks for processed data
        self.rating_callback: Optional[Callable] = None
        self.interaction_callback: Optional[Callable] = None
    
    def start_processing(self):
        """Start the background processing thread."""
        if not self.is_running:
            self.is_running = True
            self.processing_thread = threading.Thread(target=self._process_events)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            self.logger.info("Started streaming data processing")
    
    def stop_processing(self):
        """Stop the background processing thread."""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        self.logger.info("Stopped streaming data processing")
    
    def _process_events(self):
        """Background thread for processing streaming events."""
        while self.is_running:
            try:
                # Process events from queue
                events_processed = 0
                while events_processed < self.batch_size and self.is_running:
                    try:
                        event = self.event_queue.get(timeout=1)
                        self._handle_event(event)
                        events_processed += 1
                    except Empty:
                        break
                
                # Check if it's time to flush buffers
                if self._should_flush():
                    self._flush_buffers()
                    
                time.sleep(0.1)  # Prevent tight loop
                
            except Exception as e:
                self.logger.error(f"Error in event processing: {e}")
                time.sleep(1)
    
    def _handle_event(self, event: Dict[str, Any]):
        """Handle individual streaming event."""
        event_type = event.get('type')
        timestamp = datetime.fromisoformat(event.get('timestamp', datetime.now().isoformat()))
        
        if event_type == 'rating':
            self._process_rating_event(event, timestamp)
        elif event_type == 'interaction':
            self._process_interaction_event(event, timestamp)
        else:
            self.logger.warning(f"Unknown event type: {event_type}")
    
    def _process_rating_event(self, event: Dict[str, Any], timestamp: datetime):
        """Process a user rating event."""
        rating_data = {
            'user_id': event.get('user_id'),
            'movie_id': event.get('movie_id'),
            'rating': event.get('rating', 0.0),
            'timestamp': timestamp,
            'source': event.get('source', 'streaming'),
            'session_id': event.get('session_id'),
            'device_type': event.get('device_type', 'unknown')
        }
        
        # Validate rating
        if self._validate_rating(rating_data):
            self.rating_buffer.append(rating_data)
            self.logger.debug(f"Added rating to buffer: {rating_data['user_id']} -> {rating_data['movie_id']}")
        else:
            self.logger.warning(f"Invalid rating event: {rating_data}")
    
    def _process_interaction_event(self, event: Dict[str, Any], timestamp: datetime):
        """Process a user interaction event."""
        interaction_data = {
            'user_id': event.get('user_id'),
            'movie_id': event.get('movie_id'),
            'interaction_type': event.get('interaction_type'),  # view, click, share, etc.
            'timestamp': timestamp,
            'duration': event.get('duration', 0),
            'source': event.get('source', 'streaming'),
            'session_id': event.get('session_id'),
            'metadata': event.get('metadata', {})
        }
        
        if self._validate_interaction(interaction_data):
            self.interaction_buffer.append(interaction_data)
            self.logger.debug(f"Added interaction to buffer: {interaction_data['user_id']} -> {interaction_data['interaction_type']}")
        else:
            self.logger.warning(f"Invalid interaction event: {interaction_data}")
    
    def _validate_rating(self, rating_data: Dict[str, Any]) -> bool:
        """Validate rating event data."""
        required_fields = ['user_id', 'movie_id', 'rating']
        
        # Check required fields
        for field in required_fields:
            if not rating_data.get(field):
                return False
        
        # Validate rating range
        rating = rating_data.get('rating', 0.0)
        if not isinstance(rating, (int, float)) or rating < 0.5 or rating > 5.0:
            return False
            
        return True
    
    def _validate_interaction(self, interaction_data: Dict[str, Any]) -> bool:
        """Validate interaction event data."""
        required_fields = ['user_id', 'interaction_type']
        
        # Check required fields
        for field in required_fields:
            if not interaction_data.get(field):
                return False
        
        # Validate interaction type
        valid_types = ['view', 'click', 'share', 'like', 'bookmark', 'search']
        if interaction_data.get('interaction_type') not in valid_types:
            return False
            
        return True
    
    def _should_flush(self) -> bool:
        """Check if buffers should be flushed."""
        time_to_flush = datetime.now() - self.last_flush_time > timedelta(seconds=self.flush_interval)
        buffer_full = len(self.rating_buffer) >= self.batch_size or len(self.interaction_buffer) >= self.batch_size
        
        return time_to_flush or buffer_full
    
    def _flush_buffers(self):
        """Flush buffered data to callbacks."""
        try:
            # Flush rating buffer
            if self.rating_buffer and self.rating_callback:
                ratings_df = pd.DataFrame(list(self.rating_buffer))
                self.rating_callback(ratings_df)
                self.rating_buffer.clear()
                self.logger.info(f"Flushed {len(ratings_df)} ratings")
            
            # Flush interaction buffer
            if self.interaction_buffer and self.interaction_callback:
                interactions_df = pd.DataFrame(list(self.interaction_buffer))
                self.interaction_callback(interactions_df)
                self.interaction_buffer.clear()
                self.logger.info(f"Flushed {len(interactions_df)} interactions")
            
            self.last_flush_time = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error flushing buffers: {e}")
    
    def add_event(self, event: Dict[str, Any]):
        """
        Add streaming event to processing queue.
        
        Args:
            event: Event dictionary with type, user_id, and other data
        """
        if not event.get('timestamp'):
            event['timestamp'] = datetime.now().isoformat()
            
        self.event_queue.put(event)
    
    def add_rating_event(self, user_id: str, movie_id: str, rating: float, 
                        session_id: Optional[str] = None, **kwargs):
        """
        Add a rating event to the stream.
        
        Args:
            user_id: ID of the user
            movie_id: ID of the movie
            rating: Rating value (0.5-5.0)
            session_id: Optional session identifier
            **kwargs: Additional metadata
        """
        event = {
            'type': 'rating',
            'user_id': user_id,
            'movie_id': movie_id,
            'rating': rating,
            'session_id': session_id,
            **kwargs
        }
        self.add_event(event)
    
    def add_interaction_event(self, user_id: str, interaction_type: str, 
                            movie_id: Optional[str] = None, duration: int = 0,
                            session_id: Optional[str] = None, **kwargs):
        """
        Add an interaction event to the stream.
        
        Args:
            user_id: ID of the user
            interaction_type: Type of interaction (view, click, etc.)
            movie_id: Optional movie ID for movie-specific interactions
            duration: Duration of interaction in seconds
            session_id: Optional session identifier
            **kwargs: Additional metadata
        """
        event = {
            'type': 'interaction',
            'user_id': user_id,
            'interaction_type': interaction_type,
            'movie_id': movie_id,
            'duration': duration,
            'session_id': session_id,
            **kwargs
        }
        self.add_event(event)
    
    def set_rating_callback(self, callback: Callable[[pd.DataFrame], None]):
        """Set callback function for processed rating data."""
        self.rating_callback = callback
    
    def set_interaction_callback(self, callback: Callable[[pd.DataFrame], None]):
        """Set callback function for processed interaction data."""
        self.interaction_callback = callback
    
    def get_buffer_stats(self) -> Dict[str, Any]:
        """Get current buffer statistics."""
        return {
            'rating_buffer_size': len(self.rating_buffer),
            'interaction_buffer_size': len(self.interaction_buffer),
            'event_queue_size': self.event_queue.qsize(),
            'is_running': self.is_running,
            'last_flush_time': self.last_flush_time.isoformat(),
            'buffer_utilization': {
                'rating': len(self.rating_buffer) / self.buffer_size,
                'interaction': len(self.interaction_buffer) / self.buffer_size
            }
        }
    
    def get_recent_ratings(self, limit: int = 100) -> pd.DataFrame:
        """Get recent rating events as DataFrame."""
        recent_ratings = list(self.rating_buffer)[-limit:] if self.rating_buffer else []
        return pd.DataFrame(recent_ratings)
    
    def get_recent_interactions(self, limit: int = 100) -> pd.DataFrame:
        """Get recent interaction events as DataFrame."""
        recent_interactions = list(self.interaction_buffer)[-limit:] if self.interaction_buffer else []
        return pd.DataFrame(recent_interactions)