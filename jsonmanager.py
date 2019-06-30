import json
import os.path
import sys
import logging

class JsonManager:
    def __init__(self, filename):
        try:
            filer = open(filename, "r")
        except FileNotFoundError:
            filer = open(filename, "w")
            filer.write('{}')
            filer.close()
            filer = open(filename, "r")
        except EnvironmentError:
            logging.error("Error opening counter for reading.")
            logging.error(sys.exc_info()[0])
            return

        self.data = {}
        try:
            if filer:
                self.data = json.load(filer)
                filer.close()
        except json.JSONDecodeError as e:
            logging.error("JSON parse error: " + e.msg)
            return
        
        self.filename = filename

    def get(self, key):
        if key in self.data:
            return self.data[key]
        else:
            logging.error("Key is not on the counter.")
            return None

    def set(self, key, value):
        self.data[key] = value

        try:
            file = open(self.filename, "w")
        except EnvironmentError:
            logging.error("Error opening counter for writing.")
            logging.error(sys.exc_info()[0])
            return False

        try:
            json.dump(self.data, file)
            file.close()
            return True
        except TypeError:
            logging.error("Error saving to file.")
            return False