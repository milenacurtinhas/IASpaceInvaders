import numpy as np
from typing import Tuple

class BatAlgorithm:
    def __init__(self, population_size: int, weight_dim: int, fmin: float = 0, 
                 fmax: float = 2, alpha: float = 0.9, gamma: float = 0.9):
        self.population_size = population_size
        self.weight_dim = weight_dim
        self.fmin = fmin
        self.fmax = fmax
        self.alpha = alpha
        self.gamma = gamma
        self.population = self._initialize_population()
        
        # Parâmetros persistentes (conforme artigo)
        self.velocities = np.zeros_like(self.population)
        self.frequencies = np.zeros(self.population_size)
        self.loudness = np.random.uniform(1, 2, self.population_size)  # A0
        self.pulse_rate = np.random.uniform(0, 1, self.population_size)  # r0
        self.initial_pulse_rate = self.pulse_rate.copy()  # Para Eq. 6
        self.iteration = 0
    
    def _initialize_population(self) -> np.ndarray:
        return np.random.uniform(-1, 1, (self.population_size, self.weight_dim))
    
    def evolve(self, fitness_function, parallel=False) -> Tuple[np.ndarray, float]:
        """Executa uma iteração do algoritmo dos morcegos conforme Yang (2010)"""
        
        # Avaliação da população
        if parallel:
            fitness_scores = np.array(fitness_function(self.population))
        else:
            fitness_scores = np.array([fitness_function(individual) for individual in self.population])
        
        best_idx = np.argmax(fitness_scores)
        best_bat = self.population[best_idx].copy()
        best_fitness = fitness_scores[best_idx]
        
        # Algoritmo dos morcegos conforme equações do artigo
        for i in range(self.population_size):
            # Eq. 2: Atualiza frequência
            beta = np.random.uniform(0, 1)
            self.frequencies[i] = self.fmin + (self.fmax - self.fmin) * beta
            
            # Eq. 3 e 4: Atualiza velocidade e posição
            self.velocities[i] = self.velocities[i] + (self.population[i] - best_bat) * self.frequencies[i]
            new_bat = self.population[i] + self.velocities[i]
            
            # Eq. 5: Busca local (quando rand > ri)
            if np.random.rand() > self.pulse_rate[i]:
                epsilon = np.random.uniform(-1, 1, self.weight_dim)
                avg_loudness = np.mean(self.loudness)
                new_bat = best_bat + epsilon * avg_loudness
            
            # Avaliação da nova solução
            if parallel:
                new_fitness = fitness_function([new_bat])[0]
            else:
                new_fitness = fitness_function(new_bat)
            
            # Critério de aceitação (rand < Ai AND fitness melhor)
            if np.random.rand() < self.loudness[i] and new_fitness > fitness_scores[i]:
                self.population[i] = new_bat
                fitness_scores[i] = new_fitness
                
                # Eq. 6: Atualiza parâmetros apenas quando aceita solução
                self.loudness[i] *= self.alpha  # A^(t+1)_i = α × A^t_i
                self.pulse_rate[i] = self.initial_pulse_rate[i] * (1 - np.exp(-self.gamma * self.iteration))
            
            # Atualiza melhor global
            if new_fitness > best_fitness:
                best_fitness = new_fitness
                best_bat = new_bat.copy()
        
        # Incrementa contador de iteração
        self.iteration += 1
        
        return best_bat, best_fitness