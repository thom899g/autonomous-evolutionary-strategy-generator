"""
Neuroevolution configuration parameters for trading strategy evolution.
Architectural choice: Centralized configuration to ensure reproducibility
and easy hyperparameter tuning across the entire ecosystem.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple
import logging

@dataclass
class NeuroEvolutionConfig:
    """Configuration for neuroevolution parameters"""
    
    # Population parameters
    POPULATION_SIZE: int = 100
    ELITE_SIZE: int = 10
    GENERATION_COUNT: int = 50
    MUTATION_RATE: float = 0.15
    CROSSOVER_RATE: float = 0.70
    
    # Genome representation
    GENE_BOUNDS: Dict[str, Tuple[float, float]] = {
        # Technical indicator parameters
        'rsi_period': (5, 30),
        'sma_short': (5, 50),
        'sma_long': (20, 200),
        'bb_std': (1.5, 3.0),
        'atr_period': (10, 30),
        'volume_sma': (10, 30),
        # Risk parameters
        'stop_loss_pct': (0.01, 0.05),
        'take_profit_pct': (0.02, 0.10),
        'position_size': (0.01, 0.10),
        # Threshold parameters
        'buy_threshold': (0.5, 0.9),
        'sell_threshold': (0.5, 0.9)
    }
    
    # Fitness function weights
    FITNESS_WEIGHTS: Dict[str, float] = {
        'sharpe_ratio': 2.0,
        'max_drawdown': -1.5,
        'profit_factor': 1.5,
        'win_rate': 1.0,
        'total_return': 1.0,
        'calmar_ratio': 1.5
    }
    
    # Evolution constraints
    MIN_TRADES_FOR_EVALUATION: int = 10
    MAX_GENE_AGE: int = 20  # generations
    
    # Market parameters
    SYMBOLS: List[str] = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
    TIMEFRAMES: List[str] = ['1h', '4h', '1d']
    TRAINING_PERIOD_DAYS: int = 90
    VALIDATION_PERIOD_DAYS: int = 30
    
    # Firestore collections
    FIRESTORE_COLLECTIONS: Dict[str, str] = {
        'population': 'neuroevolution_population',
        'performance': 'strategy_performance',
        'market_data': 'market_ohlcv',
        'transactions': 'evolution_transactions'
    }
    
    @classmethod
    def from_env(cls):
        """Load configuration from environment variables with fallbacks"""
        config = cls()
        
        # Override from environment if present
        population_size = os.getenv('EVOLUTION_POPULATION_SIZE')
        if population_size:
            config.POPULATION_SIZE = int(population_size)
            
        mutation_rate = os.getenv('EVOLUTION_MUTATION_RATE')
        if mutation_rate:
            config.MUTATION_RATE = float(mutation_rate)
            
        return config
    
    def validate(self) -> bool:
        """Validate configuration parameters"""
        try:
            if self.POPULATION_SIZE <= 0:
                raise ValueError("Population size must be positive")
            if self.ELITE_SIZE >= self.POPULATION_SIZE:
                raise ValueError("Elite size must be less than population size")
            if not 0 <= self.MUTATION_RATE <= 1:
                raise ValueError("Mutation rate must be between 0 and 1")
            if not 0 <= self.CROSSOVER_RATE <= 1:
                raise ValueError("Crossover rate must be between 0 and 1")
            return True
        except ValueError as e:
            logging.error(f"Configuration validation failed: {e}")
            return False