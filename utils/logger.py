import sys
import os

class Logger(object):
    def __init__(self, path, filename="output.log"):
        self.terminal = sys.stdout
        self.log = open(os.path.join(path, filename), "a", buffering=1)  # Ensure line buffering
        sys.stdout = self  # Redirect sys.stdout to the same logger
        sys.stderr = self  # Redirect sys.stderr to the same logger

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()