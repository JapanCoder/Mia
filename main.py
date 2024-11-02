import sys
import os
import ella  # Importing Ella's core functionalities
import threading

sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

from config import settings
from core.voice_interface import VoiceInterface

def main():
    # Initialize Ella and Hack
    ella_instance = ella.Ella()  # Initialize the Ella instance from the updated ella.py
    hack_instance = hack.Hack()  # Initialize Hack instance from updated hack.py
    
    # Retrieve voice settings from the configuration
    voice_language = settings.VOICE_SETTINGS['language']
    voice_gender = settings.VOICE_SETTINGS['gender']
    
    # Initialize the voice interface with the language and gender settings
    voice_interface = VoiceInterface(voice_language, voice_gender)
    voice_interface.speak("Hello, I'm Ella, your personal assistant. How can I assist you today?")
    
    # Run Hack monitoring on a separate thread
    monitoring_thread = threading.Thread(target=hack_instance.silent_monitor)
    monitoring_thread.daemon = True  # Marking the thread as a daemon, so it exits when main exits
    monitoring_thread.start()

    # Main loop for interaction
    while True:
        # Get user input through voice
        user_input = voice_interface.listen()
        
        # Check if valid input was received
        if not isinstance(user_input, str) or user_input.strip() == "":
            voice_interface.speak("Sorry, I didn't catch that. Could you please repeat?")
            continue
        
        # Engage Ella's conversation module based on user input
        ella_response = ella_instance.interact("User", user_input)
        voice_interface.speak(ella_response)
        
        # Handle camera monitoring commands
        if "start camera" in user_input.lower():
            ella_instance.advanced_ai.singing_module.start_camera_monitoring()
            voice_interface.speak("Camera monitoring started.")
        
        elif "stop camera" in user_input.lower():
            ella_instance.advanced_ai.singing_module.stop_camera_monitoring()
            voice_interface.speak("Camera monitoring stopped.")
        
        # Manage FM signals based on commands
        if "start FM" in user_input.lower():
            hack_instance.start_fm_communication()
            voice_interface.speak("FM communication initiated.")
        
        elif "stop FM" in user_input.lower():
            hack_instance.stop_fm_communication()
            voice_interface.speak("FM communication terminated.")
        
        # Handle singing command
        if "sing a song" in user_input.lower():
            ella_instance.sing()

if __name__ == "__main__":
    main()
