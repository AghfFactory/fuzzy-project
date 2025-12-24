"""
Fuzzy logic engine using scikit-fuzzy for speaker recognition.
Uses Mamdani inference with membership functions for each feature.
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from typing import Dict, List, Optional
import logging

from database import Database, UserProfile

logger = logging.getLogger(__name__)


class FuzzyEngine:
    """Fuzzy logic engine for speaker recognition."""
    
    def __init__(self):
        self.db: Optional[Database] = None
        self.user_profiles: Dict[str, Dict] = {}
        self.fuzzy_systems: Dict[str, ctrl.ControlSystem] = {}
        self.simulators: Dict[str, ctrl.ControlSystemSimulation] = {}
        
        # Define universal input ranges (will be adjusted per user)
        self.mfcc_range = np.arange(-500, 500, 1)
        self.centroid_range = np.arange(0, 8000, 10)
        self.zcr_range = np.arange(0, 0.5, 0.001)
        
        # Output confidence range
        self.confidence_range = np.arange(0, 1, 0.01)
    
    def initialize(self, database: Database):
        """Initialize fuzzy engine with database connection."""
        self.db = database
        self._load_user_profiles()
        logger.info(f"Fuzzy engine initialized with {len(self.user_profiles)} users")
    
    def _load_user_profiles(self):
        """Load all user profiles from database and create fuzzy systems."""
        if not self.db:
            return
        
        users = self.db.get_all_users()
        self.user_profiles = {}
        self.fuzzy_systems = {}
        self.simulators = {}
        
        for user in users:
            self._create_fuzzy_system_for_user(user)
        
        logger.info(f"Loaded {len(users)} user profiles")
    
    def _create_fuzzy_system_for_user(self, user: UserProfile):
        """Create fuzzy membership functions and inference system for a user."""
        name = user.name
        
        # Store user profile
        self.user_profiles[name] = {
            'mfcc_mean': user.mfcc_mean,
            'spectral_centroid': user.spectral_centroid,
            'zero_crossing_rate': user.zero_crossing_rate
        }
        
        # Define input variables with membership functions
        # MFCC1 mean
        mfcc_input = ctrl.Antecedent(self.mfcc_range, 'mfcc_mean')
        mfcc_input['low'] = fuzz.trimf(
            self.mfcc_range,
            [user.mfcc_mean - 200, user.mfcc_mean - 100, user.mfcc_mean]
        )
        mfcc_input['medium'] = fuzz.trimf(
            self.mfcc_range,
            [user.mfcc_mean - 100, user.mfcc_mean, user.mfcc_mean + 100]
        )
        mfcc_input['high'] = fuzz.trimf(
            self.mfcc_range,
            [user.mfcc_mean, user.mfcc_mean + 100, user.mfcc_mean + 200]
        )
        
        # Spectral centroid
        centroid_input = ctrl.Antecedent(self.centroid_range, 'spectral_centroid')
        centroid_input['low'] = fuzz.trimf(
            self.centroid_range,
            [max(0, user.spectral_centroid - 2000), user.spectral_centroid - 1000, user.spectral_centroid]
        )
        centroid_input['medium'] = fuzz.trimf(
            self.centroid_range,
            [user.spectral_centroid - 1000, user.spectral_centroid, user.spectral_centroid + 1000]
        )
        centroid_input['high'] = fuzz.trimf(
            self.centroid_range,
            [user.spectral_centroid, user.spectral_centroid + 1000, min(8000, user.spectral_centroid + 2000)]
        )
        
        # Zero-crossing rate
        zcr_input = ctrl.Antecedent(self.zcr_range, 'zero_crossing_rate')
        zcr_input['low'] = fuzz.trimf(
            self.zcr_range,
            [max(0, user.zero_crossing_rate - 0.1), user.zero_crossing_rate - 0.05, user.zero_crossing_rate]
        )
        zcr_input['medium'] = fuzz.trimf(
            self.zcr_range,
            [user.zero_crossing_rate - 0.05, user.zero_crossing_rate, user.zero_crossing_rate + 0.05]
        )
        zcr_input['high'] = fuzz.trimf(
            self.zcr_range,
            [user.zero_crossing_rate, user.zero_crossing_rate + 0.05, min(0.5, user.zero_crossing_rate + 0.1)]
        )
        
        # Output variable (confidence)
        confidence_output = ctrl.Consequent(self.confidence_range, 'confidence')
        confidence_output['low'] = fuzz.trimf(self.confidence_range, [0, 0, 0.3])
        confidence_output['medium'] = fuzz.trimf(self.confidence_range, [0.2, 0.5, 0.8])
        confidence_output['high'] = fuzz.trimf(self.confidence_range, [0.7, 1.0, 1.0])
        
        # Define rules (Mamdani inference)
        # Rule 1: If all features match → high confidence
        rule1 = ctrl.Rule(
            mfcc_input['medium'] & centroid_input['medium'] & zcr_input['medium'],
            confidence_output['high']
        )
        
        # Rule 2: If two features match → medium confidence
        rule2a = ctrl.Rule(
            (mfcc_input['medium'] | mfcc_input['low'] | mfcc_input['high']) &
            (centroid_input['medium'] | centroid_input['low'] | centroid_input['high']) &
            ~(mfcc_input['medium'] & centroid_input['medium'] & zcr_input['medium']),
            confidence_output['medium']
        )
        
        # Rule 3: If features don't match → low confidence
        rule3 = ctrl.Rule(
            ~(mfcc_input['medium'] | mfcc_input['low'] | mfcc_input['high']) |
            ~(centroid_input['medium'] | centroid_input['low'] | centroid_input['high']) |
            ~(zcr_input['medium'] | zcr_input['low'] | zcr_input['high']),
            confidence_output['low']
        )
        
        # Create control system
        fuzzy_system = ctrl.ControlSystem([rule1, rule2a, rule3])
        simulator = ctrl.ControlSystemSimulation(fuzzy_system)
        
        self.fuzzy_systems[name] = fuzzy_system
        self.simulators[name] = simulator
        
        logger.debug(f"Created fuzzy system for user: {name}")
    
    def add_user_profile(self, user_id: int, name: str, features: Dict[str, float]):
        """Add or update a user profile in the fuzzy engine."""
        if not self.db:
            return
        
        user = self.db.get_user_profile(user_id)
        if user:
            self._create_fuzzy_system_for_user(user)
            logger.info(f"Added fuzzy system for user: {name}")
    
    def recognize_speaker(self, features: Dict[str, float]) -> Dict[str, any]:
        """
        Recognize speaker using fuzzy inference.
        
        Args:
            features: Dictionary with mfcc_mean, spectral_centroid, zero_crossing_rate
        
        Returns:
            Dictionary with 'speaker' (name or 'Unknown') and 'confidence' (0-1)
        """
        if len(self.user_profiles) == 0:
            return {"speaker": "Unknown", "confidence": 0.0}
        
        best_match = None
        best_confidence = 0.0
        
        # Test against each registered user
        for name, simulator in self.simulators.items():
            try:
                # Set input values
                simulator.input['mfcc_mean'] = features['mfcc_mean']
                simulator.input['spectral_centroid'] = features['spectral_centroid']
                simulator.input['zero_crossing_rate'] = features['zero_crossing_rate']
                
                # Compute
                simulator.compute()
                
                # Get output confidence
                confidence = float(simulator.output['confidence'])
                
                # Clamp confidence to [0, 1]
                confidence = max(0.0, min(1.0, confidence))
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = name
                    
            except Exception as e:
                logger.error(f"Fuzzy inference error for {name}: {str(e)}")
                continue
        
        # If confidence is too low, return Unknown
        if best_confidence < 0.5:
            return {"speaker": "Unknown", "confidence": best_confidence}
        
        return {"speaker": best_match, "confidence": best_confidence}

