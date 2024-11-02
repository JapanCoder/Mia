import speech_recognition as sr
import pyttsx3

class VoiceInterface:
    def __init__(self, language='en', gender='female', ella_instance=None):
        self.recognizer = sr.Recognizer()
        self.language = language
        self.gender = gender
        self.ella = ella_instance  # Connect to the Ella instance
        self.engine = pyttsx3.init()
        
        voices = self.engine.getProperty('voices')
        if voices:
            if gender == 'male' and len(voices) > 0:
                self.engine.setProperty('voice', voices[0].id)
            elif gender == 'female' and len(voices) > 1:
                self.engine.setProperty('voice', voices[1].id)
            else:
                print("Selected gender voice not found, using default voice.")
        else:
            print("No voices found on this system.")
        
    def listen(self):
        with sr.Microphone() as source:
            print("Listening...")
            self.recognizer.adjust_for_ambient_noise(source)  # Optional: Adjust for background noise
            audio = self.recognizer.listen(source)
        try:
            user_input = self.recognizer.recognize_google(audio, language=self.language)
            print(f"User said: {user_input}")
            return user_input
        except sr.UnknownValueError:
            print("Sorry, I didn't catch that.")
            return ""
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
            return ""

    def speak(self, text):
        print(f"Speaking: {text}")
        self.engine.say(text)
        self.engine.runAndWait()

    def change_voice(self, gender):
        """Change voice between male and female."""
        voices = self.engine.getProperty('voices')
        if gender == 'male' and len(voices) > 0:
            self.engine.setProperty('voice', voices[0].id)
        elif gender == 'female' and len(voices) > 1:
            self.engine.setProperty('voice', voices[1].id)
        else:
            print("Selected gender voice not found.")

    def set_speaking_rate(self, rate):
        """Change the speaking rate for the voice engine."""
        self.engine.setProperty('rate', rate)

    def set_volume(self, volume):
        """Set the volume of the voice engine."""
        if 0 <= volume <= 1:
            self.engine.setProperty('volume', volume)
        else:
            print("Volume must be between 0 and 1.")
    
    def interact(self, username, user_input):
        """Interface method for interacting with Ella."""
        if self.ella:
            spoken_text = self.listen()  # Use listen method
            response = self.ella.handle_user_input(username, spoken_text)  # Pass to Ella for a response
            self.speak(response)  # Use speak method to give a response
            return response
        else:
            print("Ella instance not connected.")
            return None