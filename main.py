import sys
import os
import subprocess
import eel

# Add the directory containing the 'engine' module to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'engine')))

from engine.features import playAssistantSound
from engine.command import speak
from engine.auth import recoganize

def start():
    try:
        # Initialize eel with the web directory
        eel.init("www")

        # Play assistant sound
        playAssistantSound()

        @eel.expose
        def init():
            try:
                subprocess.call([r'device.bat'])
                eel.hideLoader()
                speak("Ready for Face Authentication")
                flag = recoganize.AuthenticateFace()
                if flag == 1:
                    eel.hideFaceAuth()
                    speak("Face Authentication Successful")
                    eel.hideFaceAuthSuccess()
                    speak("Hello, Welcome Sir, How can I help You")
                    eel.hideStart()
                    playAssistantSound()
                else:
                    speak("Face Authentication Failed")
            except Exception as e:
                speak(f"An error occurred: {e}")

        # Open the web interface
        os.system('start msedge.exe --app="http://127.0.0.1:5500/www/intex.html"')
        
        # Start eel with the specified HTML file
        eel.start('index.html', mode=None, host='localhost', block=True)
    except KeyboardInterrupt:
        print("Script interrupted by user")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    start()