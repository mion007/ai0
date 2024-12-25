import multiprocessing
import subprocess
import signal
import sys

# To run Jarvis
def startJarvis():
    # Code for process 1
    print("Process 1 is running.")
    from main import start
    start()

# To run hotword
def listenHotword():
    # Code for process 2
    print("Process 2 is running.")
    from engine.features import hotword
    hotword()

def handle_interrupt(signal, frame):
    print("Interrupted, terminating processes...")
    p1.terminate()
    p2.terminate()
    p1.join()
    p2.join()
    sys.exit(0)

# Start both processes
if __name__ == '__main__':
    p1 = multiprocessing.Process(target=startJarvis)
    p2 = multiprocessing.Process(target=listenHotword)
    
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, handle_interrupt)
    signal.signal(signal.SIGTERM, handle_interrupt)
    
    try:
        p1.start()
        p2.start()
        
        p1.join()
        
        if p2.is_alive():
            p2.terminate()
            p2.join()
    
        print("System stopped.")
        
    except KeyboardInterrupt:
        print("Keyboard interrupt received, terminating processes...")
        p1.terminate()
        p2.terminate()
        p1.join()
        p2.join()
        print("Processes terminated.")
    except Exception as e:
        print(f"An error occurred: {e}")
        p1.terminate()
        p2.terminate()
        p1.join()
        p2.join()
        sys.exit(1)