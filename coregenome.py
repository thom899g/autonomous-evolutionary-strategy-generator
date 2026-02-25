"""
Genome representation for trading strategies.
Architectural choice: Object-oriented genome with validation and serialization
to enable complex genetic operations while maintaining data integrity.
"""

import numpy as np
from typing import Dict, List, Any, Optional
import hashlib
import json
from datetime import datetime, timezone
import logging
from dataclasses import dataclass, field
import uuid

@dataclass
class TradingGene:
    """Represents a single gene in the trading strategy genome"""
    name: str
    value: float
    bounds: tuple
    gene_type: str = 'continuous'  # continuous, discrete, binary
    
    def mutate(self, mutation_rate: float, mutation_strength: float = 0.1):
        """Apply mutation to gene value"""
        if np.random.random() < mutation_rate:
            if self.gene_type == 'continuous':
                # Gaussian mutation with bounds constraint
                min_val, max_val = self.bounds
                delta = np.random.normal(0, mutation_strength