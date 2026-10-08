"""
BioMorph — нейроморфные системы на основе коннектома Drosophila.

Модули:
    neuron    — LIF-нейрон
    recorder  — запись спайков (в разработке)
    synapse   — синапсы (в разработке)
    network   — сеть нейронов (в разработке)
"""

from .neuron import LIFNeuron

__all__ = ["LIFNeuron"]
__version__ = "0.1.0"
