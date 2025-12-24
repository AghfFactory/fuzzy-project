"""
Audio preprocessing and conversion module.
Converts various audio formats (M4A, 3GP, etc.) to WAV format for consistent processing.
"""

import logging
import tempfile
import os
import numpy as np
from typing import Optional, Tuple
import librosa

logger = logging.getLogger(__name__)


class AudioConverter:
    """Converts audio files to WAV format for consistent processing."""
    
    def __init__(self, target_sample_rate: int = 16000, target_channels: int = 1):
        """
        Initialize audio converter.
        
        Args:
            target_sample_rate: Target sample rate in Hz (default: 16000)
            target_channels: Target number of channels, 1=mono (default: 1)
        """
        self.target_sample_rate = target_sample_rate
        self.target_channels = target_channels
    
    def detect_format(self, audio_bytes: bytes) -> str:
        """
        Detect audio format from file signature.
        
        Args:
            audio_bytes: Raw audio file bytes
            
        Returns:
            File extension (e.g., '.m4a', '.3gp', '.wav')
        """
        if len(audio_bytes) < 4:
            return '.3gp'  # Default
        
        # Check for M4A/MP4 signature (ftyp)
        if audio_bytes[:4] == b'ftyp':
            # Check if it's M4A
            if b'm4a' in audio_bytes[:20] or b'MP4' in audio_bytes[:20] or b'isom' in audio_bytes[:20]:
                return '.m4a'
            return '.mp4'
        
        # Check for WAV signature (RIFF)
        if audio_bytes[:4] == b'RIFF' and len(audio_bytes) > 8:
            if audio_bytes[8:12] == b'WAVE':
                return '.wav'
        
        # Check for 3GP signature
        if audio_bytes[4:8] == b'ftyp' or b'3gp' in audio_bytes[:20]:
            return '.3gp'
        
        # Check for OGG signature
        if audio_bytes[:4] == b'OggS':
            return '.ogg'
        
        # Default to 3GP (common Android format)
        return '.3gp'
    
    def convert_to_wav(self, audio_bytes: bytes, input_format: Optional[str] = None) -> Tuple[np.ndarray, int]:
        """
        Convert audio bytes to WAV format and return as numpy array.
        
        Args:
            audio_bytes: Raw audio file bytes
            input_format: Optional format hint (e.g., 'm4a', '3gp'). If None, auto-detect.
            
        Returns:
            Tuple of (audio_array, sample_rate) where:
            - audio_array: numpy array of normalized audio samples [-1, 1]
            - sample_rate: sample rate in Hz
            
        Raises:
            ValueError: If audio cannot be converted
        """
        temp_input_file = None
        temp_wav_file = None
        
        try:
            # Detect format if not provided
            if input_format is None:
                file_ext = self.detect_format(audio_bytes)
                input_format = file_ext.lstrip('.')
            else:
                file_ext = f'.{input_format.lstrip(".")}'
            
            # Write input audio to temporary file (don't auto-delete, we'll handle it)
            temp_input_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext).name
            with open(temp_input_file, 'wb') as f:
                f.write(audio_bytes)
            
            # Try librosa first (works for WAV, MP3, and some other formats)
            try:
                import warnings
                with warnings.catch_warnings():
                    # Suppress all warnings from librosa (PySoundFile, FutureWarning, etc.)
                    warnings.filterwarnings("ignore", category=FutureWarning)
                    warnings.filterwarnings("ignore", category=UserWarning)
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    audio_array, sample_rate = librosa.load(
                        temp_input_file,
                        sr=self.target_sample_rate,
                        mono=(self.target_channels == 1)
                    )
                
                if audio_array is None or len(audio_array) == 0:
                    raise ValueError("librosa loaded empty audio")
                
                logger.debug(f"Successfully loaded with librosa: {len(audio_array)} samples at {sample_rate}Hz")
                return audio_array, sample_rate
                
            except Exception as librosa_error:
                logger.debug(f"librosa failed ({type(librosa_error).__name__}), trying pydub converter...")
                
                # Try pydub for format conversion (requires ffmpeg)
                try:
                    from pydub import AudioSegment
                    from pydub.utils import which
                    import shutil
                    import subprocess
                    
                    # Check if ffmpeg is available - try multiple methods
                    ffmpeg_path = None
                    ffprobe_path = None
                    
                    # Method 1: Try which() function
                    ffmpeg_path = which("ffmpeg")
                    ffprobe_path = which("ffprobe")
                    
                    # Method 2: Try subprocess to find ffmpeg in PATH
                    if not ffmpeg_path:
                        try:
                            result = subprocess.run(
                                ["where", "ffmpeg"] if os.name == 'nt' else ["which", "ffmpeg"],
                                capture_output=True,
                                text=True,
                                timeout=2
                            )
                            if result.returncode == 0 and result.stdout.strip():
                                ffmpeg_path = result.stdout.strip().split('\n')[0]
                                logger.info(f"Found ffmpeg via subprocess: {ffmpeg_path}")
                        except Exception:
                            pass
                    
                    # Method 3: Try common Windows locations
                    if not ffmpeg_path:
                        common_paths = [
                            r"C:\ffmpeg\bin\ffmpeg.exe",
                            r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
                            r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
                        ]
                        if "PROGRAMFILES" in os.environ:
                            common_paths.append(os.path.join(os.environ["PROGRAMFILES"], "ffmpeg", "bin", "ffmpeg.exe"))
                        if "PROGRAMFILES(X86)" in os.environ:
                            common_paths.append(os.path.join(os.environ["PROGRAMFILES(X86)"], "ffmpeg", "bin", "ffmpeg.exe"))
                        
                        for path in common_paths:
                            if path and os.path.exists(path):
                                ffmpeg_path = path
                                logger.info(f"Found ffmpeg at: {ffmpeg_path}")
                                break
                    
                    # Method 4: Try to run ffmpeg directly (it might be in PATH but which() failed)
                    if not ffmpeg_path:
                        try:
                            result = subprocess.run(
                                ["ffmpeg", "-version"],
                                capture_output=True,
                                timeout=2
                            )
                            if result.returncode == 0 or "ffmpeg version" in result.stderr.decode('utf-8', errors='ignore'):
                                ffmpeg_path = "ffmpeg"  # Use command name, let system find it
                                logger.info("ffmpeg is accessible via system PATH")
                        except Exception:
                            pass
                    
                    # Find ffprobe similarly
                    if not ffprobe_path:
                        if ffmpeg_path and ffmpeg_path != "ffmpeg":
                            # Try same directory as ffmpeg
                            probe_path = ffmpeg_path.replace("ffmpeg.exe", "ffprobe.exe").replace("ffmpeg", "ffprobe")
                            if os.path.exists(probe_path):
                                ffprobe_path = probe_path
                        else:
                            # Try subprocess
                            try:
                                result = subprocess.run(
                                    ["where", "ffprobe"] if os.name == 'nt' else ["which", "ffprobe"],
                                    capture_output=True,
                                    text=True,
                                    timeout=2
                                )
                                if result.returncode == 0 and result.stdout.strip():
                                    ffprobe_path = result.stdout.strip().split('\n')[0]
                            except Exception:
                                pass
                            
                            if not ffprobe_path:
                                try:
                                    result = subprocess.run(
                                        ["ffprobe", "-version"],
                                        capture_output=True,
                                        timeout=2
                                    )
                                    if result.returncode == 0 or "ffprobe version" in result.stderr.decode('utf-8', errors='ignore'):
                                        ffprobe_path = "ffprobe"
                                except Exception:
                                    pass
                    
                    # Set pydub paths if found
                    if ffmpeg_path:
                        AudioSegment.converter = ffmpeg_path
                        logger.info(f"Set pydub converter to: {ffmpeg_path}")
                    if ffprobe_path:
                        AudioSegment.ffprobe = ffprobe_path
                        logger.info(f"Set pydub ffprobe to: {ffprobe_path}")
                    
                    if not ffmpeg_path:
                        raise ValueError(
                            "ffmpeg not found. Please ensure ffmpeg is installed and accessible.\n"
                            "Try running 'ffmpeg -version' in your terminal to verify installation."
                        )
                    
                    # Load audio with pydub
                    audio_segment = AudioSegment.from_file(temp_input_file, format=input_format)
                    
                    # Convert to target format
                    if audio_segment.channels != self.target_channels:
                        audio_segment = audio_segment.set_channels(self.target_channels)
                    
                    if audio_segment.frame_rate != self.target_sample_rate:
                        audio_segment = audio_segment.set_frame_rate(self.target_sample_rate)
                    
                    # Export to WAV in memory (temporary file)
                    temp_wav_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
                    audio_segment.export(temp_wav_file, format="wav")
                    
                    # Small delay to ensure file is fully written
                    import time
                    time.sleep(0.1)
                    
                    # Load WAV with librosa (guaranteed to work)
                    audio_array, sample_rate = librosa.load(
                        temp_wav_file,
                        sr=self.target_sample_rate,
                        mono=(self.target_channels == 1)
                    )
                    
                    if audio_array is None or len(audio_array) == 0:
                        raise ValueError("pydub conversion resulted in empty audio")
                    
                    logger.debug(f"Successfully converted with pydub: {len(audio_array)} samples at {sample_rate}Hz")
                    return audio_array, sample_rate
                    
                except ImportError:
                    raise ValueError(
                        "pydub is required for audio format conversion. "
                        "Install with: pip install pydub. "
                        "Note: pydub requires ffmpeg for M4A/3GP support."
                    )
                except Exception as pydub_error:
                    raise ValueError(
                        f"Failed to convert audio with pydub: {type(pydub_error).__name__}: {str(pydub_error)}. "
                        f"Original librosa error: {type(librosa_error).__name__}: {str(librosa_error)}. "
                        f"Ensure ffmpeg is installed and in PATH for format conversion."
                    )
        
        finally:
            # Clean up temporary files with retry logic for Windows file locking
            import time
            for temp_file in [temp_input_file, temp_wav_file]:
                if temp_file and os.path.exists(temp_file):
                    # Retry deletion (Windows sometimes locks files briefly)
                    for attempt in range(3):
                        try:
                            os.unlink(temp_file)
                            break
                        except (OSError, PermissionError) as e:
                            if attempt < 2:
                                time.sleep(0.1)
                            else:
                                logger.debug(f"Could not delete temp file {temp_file}: {e}")
    
    def convert_bytes_to_wav_bytes(self, audio_bytes: bytes, input_format: Optional[str] = None) -> bytes:
        """
        Convert audio bytes to WAV format bytes (for storage/transmission).
        
        Args:
            audio_bytes: Raw audio file bytes
            input_format: Optional format hint. If None, auto-detect.
            
        Returns:
            WAV file as bytes
            
        Raises:
            ValueError: If audio cannot be converted
        """
        temp_input_file = None
        temp_wav_file = None
        
        try:
            # Detect format if not provided
            if input_format is None:
                file_ext = self.detect_format(audio_bytes)
                input_format = file_ext.lstrip('.')
            else:
                file_ext = f'.{input_format.lstrip(".")}'
            
            # Write input audio to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                tmp.write(audio_bytes)
                temp_input_file = tmp.name
            
            # Use pydub for conversion
            try:
                from pydub import AudioSegment
                
                # Load audio
                audio_segment = AudioSegment.from_file(temp_input_file, format=input_format)
                
                # Convert to target format
                if audio_segment.channels != self.target_channels:
                    audio_segment = audio_segment.set_channels(self.target_channels)
                
                if audio_segment.frame_rate != self.target_sample_rate:
                    audio_segment = audio_segment.set_frame_rate(self.target_sample_rate)
                
                # Export to WAV bytes
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as wav_tmp:
                    temp_wav_file = wav_tmp.name
                    audio_segment.export(temp_wav_file, format="wav")
                    
                    # Read WAV bytes
                    with open(temp_wav_file, 'rb') as f:
                        wav_bytes = f.read()
                    
                    return wav_bytes
                    
            except ImportError:
                raise ValueError(
                    "pydub is required for audio format conversion. "
                    "Install with: pip install pydub"
                )
            except Exception as e:
                raise ValueError(f"Failed to convert audio: {type(e).__name__}: {str(e)}")
        
        finally:
            # Clean up temporary files
            for temp_file in [temp_input_file, temp_wav_file]:
                if temp_file and os.path.exists(temp_file):
                    try:
                        os.unlink(temp_file)
                    except Exception as e:
                        logger.warning(f"Failed to delete temp file {temp_file}: {e}")

