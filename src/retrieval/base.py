import os
from abc import ABC, abstractmethod

class Indexer(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def index(self, *args, **kwargs):
        """Abstract method to index data."""
        raise NotImplementedError("Subclasses must implement this method.")
    

class Retriever(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def retrieve(self, *args, **kwargs):
        """Abstract method to retrieve data."""
        raise NotImplementedError("Subclasses must implement this method.")
    
    