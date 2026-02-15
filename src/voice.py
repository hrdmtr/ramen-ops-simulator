"""
Voice output module for simulator.
Provides text-to-speech functionality for AI instructions.
"""

import subprocess
import platform
from typing import Optional


class VoiceOutput:
    """Text-to-speech output for AI instructions."""

    def __init__(self, enabled: bool = True, voice_type: str = "system", speech_rate: int = 200):
        """
        Initialize voice output.

        Args:
            enabled: Whether voice output is enabled
            voice_type: Type of voice system to use
                - "system": Use system TTS (macOS 'say' command)
                - "pyttsx3": Use pyttsx3 library (cross-platform)
                - "none": Disable voice output
            speech_rate: Speech rate in words per minute (default: 200, macOS default is ~175)
                - 200: Normal speed
                - 400: 2x speed
                - 600: 3x speed
                - 1000: ~5x speed
                - 2000: ~10x speed (maximum practical speed)
        """
        self.enabled = enabled
        self.voice_type = voice_type
        self.system = platform.system()
        self.speech_rate = speech_rate

        # Initialize pyttsx3 if needed
        self.engine = None
        if voice_type == "pyttsx3" and enabled:
            try:
                import pyttsx3
                self.engine = pyttsx3.init()
                # Set Japanese voice if available
                voices = self.engine.getProperty('voices')
                for voice in voices:
                    if 'japan' in voice.name.lower() or 'kyoko' in voice.name.lower():
                        self.engine.setProperty('voice', voice.id)
                        break
            except ImportError:
                print("⚠️ pyttsx3 not installed. Install with: pip install pyttsx3")
                self.enabled = False

    def speak(self, text: str, blocking: bool = False):
        """
        Speak text using TTS.

        Args:
            text: Text to speak
            blocking: If True, wait for speech to complete
        """
        if not self.enabled:
            return

        if self.voice_type == "system" and self.system == "Darwin":
            # macOS 'say' command
            self._speak_macos(text, blocking)
        elif self.voice_type == "pyttsx3" and self.engine:
            # pyttsx3 library
            self._speak_pyttsx3(text, blocking)
        else:
            # Fallback: print to console
            print(f"🔊 {text}")

    def _speak_macos(self, text: str, blocking: bool):
        """Speak using macOS 'say' command."""
        try:
            # Use Kyoko voice for Japanese with custom speech rate
            # -r rate: words per minute (default ~175, max practical ~2000)
            if blocking:
                subprocess.run(["say", "-v", "Kyoko", "-r", str(self.speech_rate), text])
            else:
                subprocess.Popen(["say", "-v", "Kyoko", "-r", str(self.speech_rate), text])
        except Exception as e:
            print(f"⚠️ Voice output failed: {e}")
            print(f"🔊 {text}")

    def _speak_pyttsx3(self, text: str, blocking: bool):
        """Speak using pyttsx3 library."""
        try:
            # Set speech rate for pyttsx3
            self.engine.setProperty('rate', self.speech_rate)
            self.engine.say(text)
            if blocking:
                self.engine.runAndWait()
            else:
                # Non-blocking: run in separate thread
                import threading
                thread = threading.Thread(target=self.engine.runAndWait)
                thread.start()
        except Exception as e:
            print(f"⚠️ Voice output failed: {e}")
            print(f"🔊 {text}")

    def _get_message(self, action_name: str) -> str:
        """
        Get message text for an action.

        Args:
            action_name: Action name (e.g., "BOIL_HELP", "DISH_WASH")

        Returns:
            Message text in Japanese
        """
        messages = {
            "DO_NOTHING": "待機します",
            "BOIL_HELP": "茹で場のヘルプをお願いします",
            "PLATE_HELP": "盛り付けのヘルプをお願いします",
            "SERVE_HELP": "配膳のヘルプをお願いします",
            "DISH_WASH": "皿洗いをお願いします",
        }
        return messages.get(action_name, f"{action_name}をお願いします")

    def announce_instruction(self, action_name: str, staff_name: str = None, blocking: bool = True, state: dict = None):
        """
        Announce AI instruction in natural Japanese with specific details.

        Args:
            action_name: Action name (e.g., "BOIL_HELP", "DISH_WASH")
            staff_name: Staff member name (e.g., "Aさん", "Bさん", "Cさん")
            blocking: If True, wait for speech to complete before returning
            state: Current simulator state for generating specific instructions
        """
        # Generate specific message based on state
        if state and action_name == "BOIL_HELP":
            # Calculate how many portions need to be boiled
            waiting_customers = state.get("waiting_customers", 0)
            waiting_for_food = state.get("customers_waiting_for_food", 0)
            boil_wait = state.get("boil_wait", 0)
            boil_in_progress = state.get("boil_in_progress", 0)

            # Need to boil for: waiting customers + seated customers waiting for food
            # Already being processed: boil_wait + boil_in_progress
            needed = waiting_customers + waiting_for_food
            in_pipeline = boil_wait + boil_in_progress
            shortage = max(0, needed - in_pipeline)

            # Determine how many to boil (suggest 2-4 portions at a time)
            if shortage >= 4:
                boil_count = 4
            elif shortage >= 2:
                boil_count = max(2, min(4, shortage))
            else:
                boil_count = 2  # Default to 2

            base_message = f"{boil_count}玉茹でてください"

        elif state and action_name == "DISH_WASH":
            # Check how many dirty dishes
            dirty_dishes = state.get("dirty_dishes", 0)
            clean_dishes = state.get("clean_dishes", 0)

            if dirty_dishes >= 10:
                base_message = f"皿洗いをお願いします。汚れた皿が{dirty_dishes}枚あります"
            elif clean_dishes <= 5:
                base_message = f"皿洗いをお願いします。綺麗な皿が残り{clean_dishes}枚です"
            else:
                base_message = "皿洗いをお願いします"

        else:
            # Fallback to simple message
            base_message = self._get_message(action_name)

        # Add staff name if provided
        if staff_name:
            message = f"{staff_name}、{base_message}"
        else:
            message = base_message

        self.speak(message, blocking=blocking)


# Quick test
if __name__ == "__main__":
    import time

    print("音声出力テスト")
    print("=" * 60)

    # Test system voice (macOS)
    if platform.system() == "Darwin":
        print("\n1. macOS 'say' コマンドテスト:")
        voice = VoiceOutput(enabled=True, voice_type="system")
        voice.announce_instruction("BOIL_HELP")
        time.sleep(2)
        voice.announce_instruction("DISH_WASH")
        time.sleep(2)

    # Test pyttsx3
    print("\n2. pyttsx3 テスト:")
    try:
        voice = VoiceOutput(enabled=True, voice_type="pyttsx3")
        voice.announce_instruction("PLATE_HELP")
        time.sleep(2)
        voice.announce_instruction("SERVE_HELP")
    except Exception as e:
        print(f"pyttsx3テストスキップ: {e}")

    print("\n✓ 音声出力テスト完了")
