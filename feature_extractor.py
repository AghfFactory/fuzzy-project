"""
Feature extraction using librosa for MFCC, spectral centroid, and zero-crossing rate.
"""

import librosa
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract audio features for speaker recognition."""
    
    def __init__(self):
        self.target_sample_rate = 16000
    
    def extract_features(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000
    ) -> Optional[Dict[str, float]]:
        """
        Extract MFCC, spectral centroid, and zero-crossing rate features.
        
        Args:
            audio: Audio signal as numpy array (normalized to [-1, 1])
            sample_rate: Sample rate of audio (default: 16000 Hz)
        
        Returns:
            Dictionary with features: mfcc_mean, spectral_centroid, zero_crossing_rate
        """
        try:
            # Ensure audio is not empty
            if len(audio) == 0:
                logger.warning("Empty audio signal")
                return None
            
            # Resample if needed
            if sample_rate != self.target_sample_rate:
                audio = librosa.resample(
                    audio,
                    orig_sr=sample_rate,
                    target_sr=self.target_sample_rate
                )
                sample_rate = self.target_sample_rate
            
            # Extract MFCC features (13 coefficients)
            mfccs = librosa.feature.mfcc(
                y=audio,
                sr=sample_rate,
                n_mfcc=13,
                n_fft=2048,
                hop_length=512
            )
            
            # Use first MFCC coefficient (overall spectral shape)
            mfcc_mean = np.mean(mfccs[0])
            
            # Extract spectral centroid
            spectral_centroids = librosa.feature.spectral_centroid(
                y=audio,
                sr=sample_rate,
                n_fft=2048,
                hop_length=512
            )
            spectral_centroid = np.mean(spectral_centroids)
            
            # Extract zero-crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio, frame_length=2048, hop_length=512)
            zero_crossing_rate = np.mean(zcr)
            
            features = {
                'mfcc_mean': float(mfcc_mean),
                'spectral_centroid': float(spectral_centroid),
                'zero_crossing_rate': float(zero_crossing_rate)
            }
            
            logger.debug(f"Extracted features: {features}")
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction error: {str(e)}")
            return None

