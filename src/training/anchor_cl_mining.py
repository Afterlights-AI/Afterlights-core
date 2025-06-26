from typing import List, Dict, Optional
import pandas as pd

class CLSimplePairMining:
    """
    A class for mining contrastive learning pairs from dialogue data.
    
    This class processes CSV files containing dialogue data, groups conversations
    by time windows, summarizes them, and creates positive/negative pairs for
    contrastive learning training.
    """
    
    def __init__(self):
        """Initialize the pair mining class."""
        self.df: Optional[pd.DataFrame] = None
   
    
    def create_dataset(
        self, 
        file_path: str, 
    ) -> List[Dict]:
        """
        Create dialogue datasets by text.
        
        Args:
            file_path: Path to the CSV file containing dialogue data
            
        Returns:
            List[Dict]: A list of dictionaries, each containing an 'anchor' key
                        with the text from the dialogue.
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            self.df = pd.read_csv(f)
        
        _dict = []
        for _, row in self.df.iterrows():
            text = row['text']
            _dict.append({"anchor": text})
            
        
        return _dict
    

if __name__ == "__main__":
    pass