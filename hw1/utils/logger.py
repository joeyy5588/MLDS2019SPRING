import json
import logging

logging.basicConfig(level=logging.INFO, format='')

class Logger:
    """
    Training process logger

    Note:
        Used by BaseTrainer to save training history.
    """
    def __init__(self):
        self.entries = {}

    def add_entry(self, entry):
        self.entries[len(self.entries) + 1] = entry

    def __str__(self):
        return json.dumps(self.entries, sort_keys=True, indent=4)
    
    def __len__(self):
        return len(self.entries)
        
    def __getitem__(self, key):
        if key == -1: # indicate that we want to get the last one
            return self.entries[self.__len__()]
        else:
            return self.entries[key]
    
