import numpy as np
from typing import Tuple

class BatAlgorithm:
    def __init__(self, population_size: int, weight_dim: int, fmin: float = 0, 
                 fmax: float = 1, alpha: float = 0.95, gamma: float = 0.95):  # Parâmetros mais conservadores
        self.population_size = population_size
        self.weight_dim = weight_dim
        self.fmin = fmin
        self.fmax = fmax
        self.initial_fmax = fmax  # Guarda valor inicial para adaptação
        self.alpha = alpha
        self.gamma = gamma
        self.population = self._initialize_population()
        
        # Parâmetros persistentes otimizados
        self.velocities = np.zeros_like(self.population)
        self.frequencies = np.zeros(self.population_size)
        
        # Parâmetros adaptativos mais inteligentes
        self.loudness = np.random.uniform(0.8, 1.2, self.population_size)  # A0 - range mais restrito
        self.pulse_rate = np.random.uniform(0.1, 0.3, self.population_size)  # r0 - começar baixo
        self.initial_loudness = self.loudness.copy()  # Para reset se necessário
        self.initial_pulse_rate = self.pulse_rate.copy()  # Para Eq. 6
        
        self.iteration = 0
        
        # Controle de diversidade e convergência
        self.stagnation_counter = 0
        self.last_best_fitness = -np.inf
        self.diversity_threshold = 0.01
    
    def _initialize_population(self) -> np.ndarray:
        """Inicialização Xavier/He otimizada para redes neurais"""
        population = []
        
        # Arquitetura das camadas: (input, output)
        layer_specs = [(28, 16), (17, 8), (9, 3)]
        
        for _ in range(self.population_size):
            weights = []
            
            for input_size, output_size in layer_specs:
                # Inicialização Xavier para tanh
                limit = np.sqrt(6.0 / (input_size + output_size))
                layer_weights = np.random.uniform(-limit, limit, input_size * output_size)
                weights.extend(layer_weights)
            
            population.append(np.array(weights))
        
        return np.array(population)
    
    def _calculate_diversity(self) -> float:
        """Calcula diversidade da população para controle adaptativo"""
        if self.population_size < 2:
            return 1.0
        
        # Calcula distância média entre indivíduos
        total_distance = 0
        count = 0
        
        for i in range(self.population_size):
            for j in range(i+1, self.population_size):
                distance = np.linalg.norm(self.population[i] - self.population[j])
                total_distance += distance
                count += 1
        
        return total_distance / count if count > 0 else 0.0
    
    def adaptive_parameters(self, generation: int, max_generations: int):
        """Ajusta parâmetros dinamicamente conforme progresso"""
        progress = generation / max_generations if max_generations > 0 else 0
        
        # Reduz frequência máxima ao longo do tempo (exploração → explotação)
        self.fmax = self.initial_fmax * (1 - progress * 0.5)
        
        # Ajusta alpha dinamicamente (mais conservador no final)
        self.alpha = 0.9 + 0.05 * progress
        
        # Controle de diversidade - aumenta perturbação se estagnação
        diversity = self._calculate_diversity()
        if diversity < self.diversity_threshold:
            # Adiciona perturbação para escapar de mínimos locais
            perturbation = np.random.normal(0, 0.1, self.population.shape)
            self.population += perturbation * (1 - progress)  # Menor perturbação no final
    
    def improved_local_search(self, best_bat: np.ndarray, current_bat: np.ndarray, 
                            loudness: float, generation: int = 0) -> np.ndarray:
        """Busca local melhorada com distribuição gaussiana"""
        # Usa distribuição gaussiana em vez de uniforme
        epsilon = np.random.normal(0, 0.1 * loudness, len(best_bat))
        
        # Mistura inteligente entre melhor global e exploração local
        beta = np.random.uniform(0.3, 0.7)  # Balanceamento adaptativo
        
        # Combinação: parte do melhor + parte do atual + ruído
        new_bat = beta * best_bat + (1-beta) * current_bat + epsilon
        
        # Garante que os pesos ficam dentro dos limites razoáveis
        new_bat = np.clip(new_bat, -2.0, 2.0)  # Limites mais amplos para redes neurais
        
        return new_bat
    
    def levy_flight(self, current_bat: np.ndarray, scale: float = 0.01) -> np.ndarray:
        """Implementa voo de Lévy para melhor exploração"""
        # Parâmetros do voo de Lévy
        beta = 1.5
        
        # Calcula sigma conforme distribuição de Lévy
        numerator = np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2)
        denominator = np.math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2))
        sigma = (numerator / denominator) ** (1 / beta)
        
        # Gera steps de Lévy
        u = np.random.normal(0, sigma, len(current_bat))
        v = np.random.normal(0, 1, len(current_bat))
        levy_step = u / (np.abs(v) ** (1 / beta))
        
        # Aplica o voo de Lévy
        new_bat = current_bat + scale * levy_step
        
        return np.clip(new_bat, -2.0, 2.0)
    
    def evolve(self, fitness_function, parallel=False, generation=0, max_generations=1000) -> Tuple[np.ndarray, float]:
        """Versão otimizada do algoritmo dos morcegos com melhorias adaptativas"""
        
        # Aplicar parâmetros adaptativos
        self.adaptive_parameters(generation, max_generations)
        
        # Avaliação da população
        if parallel:
            fitness_scores = np.array(fitness_function(self.population))
        else:
            fitness_scores = np.array([fitness_function(individual) for individual in self.population])
        
        best_idx = np.argmax(fitness_scores)
        best_bat = self.population[best_idx].copy()
        best_fitness = fitness_scores[best_idx]
        
        # Controle de estagnação
        if best_fitness <= self.last_best_fitness + 1e-6:  # Considera melhoria mínima
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
            self.last_best_fitness = best_fitness
        
        # Algoritmo dos morcegos otimizado
        for i in range(self.population_size):
            # Eq. 2: Atualiza frequência com variação adaptativa
            beta = np.random.uniform(0, 1)
            self.frequencies[i] = self.fmin + (self.fmax - self.fmin) * beta
            
            # Eq. 3 e 4: Atualiza velocidade e posição com amortecimento
            damping = 0.95  # Amortecimento da velocidade
            self.velocities[i] = damping * self.velocities[i] + \
                                (self.population[i] - best_bat) * self.frequencies[i]
            
            # Limita velocidade para evitar movimentos muito grandes
            velocity_limit = 0.5
            self.velocities[i] = np.clip(self.velocities[i], -velocity_limit, velocity_limit)
            
            new_bat = self.population[i] + self.velocities[i]
            
            # Eq. 5: Busca local melhorada
            if np.random.rand() > self.pulse_rate[i]:
                if self.stagnation_counter > 10:  # Se estagnado, usa Lévy flight
                    new_bat = self.levy_flight(new_bat, scale=0.01)
                else:
                    new_bat = self.improved_local_search(best_bat, self.population[i], 
                                                       self.loudness[i], generation)
            
            # Garante limites dos pesos
            new_bat = np.clip(new_bat, -2.0, 2.0)
            
            # Avaliação da nova solução
            try:
                if parallel:
                    new_fitness = fitness_function([new_bat])[0]
                else:
                    new_fitness = fitness_function(new_bat)
            except Exception as e:
                # Se houver erro na avaliação, mantém solução anterior
                new_fitness = fitness_scores[i] - 1  # Fitness pior para rejeitar
            
            # Critério de aceitação melhorado
            acceptance_probability = self.loudness[i] * np.exp(-0.1 * generation / max_generations)
            
            if (np.random.rand() < acceptance_probability and new_fitness > fitness_scores[i]) or \
               (new_fitness > best_fitness):  # Sempre aceita se é a melhor solução global
                
                self.population[i] = new_bat
                fitness_scores[i] = new_fitness
                
                # Eq. 6: Atualiza parâmetros quando aceita solução
                self.loudness[i] *= self.alpha
                
                # Pulse rate adaptativo baseado no sucesso
                improvement_factor = (new_fitness - fitness_scores[i]) / (abs(fitness_scores[i]) + 1e-8)
                pulse_increase = self.initial_pulse_rate[i] * (1 - np.exp(-self.gamma * self.iteration))
                self.pulse_rate[i] = min(0.9, pulse_increase * (1 + improvement_factor))
            
            # Atualiza melhor global
            if new_fitness > best_fitness:
                best_fitness = new_fitness
                best_bat = new_bat.copy()
        
        # Rebalanceamento populacional se necessário
        if self.stagnation_counter > 50:
            self._population_rebalancing()
            self.stagnation_counter = 0
        
        # Incrementa contador
        self.iteration += 1
        
        return best_bat, best_fitness
    
    def _population_rebalancing(self):
        """Rebalanceia população quando há estagnação prolongada"""
        # Mantém os 20% melhores, reinicializa o resto
        fitness_scores = np.array([np.random.rand() for _ in self.population])  # Placeholder
        
        num_keep = int(0.2 * self.population_size)
        best_indices = np.argsort(fitness_scores)[-num_keep:]
        
        # Cria nova população mantendo os melhores
        new_population = []
        
        # Mantém os melhores
        for idx in best_indices:
            new_population.append(self.population[idx])
        
        # Reinicializa o resto com perturbações dos melhores
        while len(new_population) < self.population_size:
            base_idx = np.random.choice(best_indices)
            perturbation = np.random.normal(0, 0.1, self.weight_dim)
            new_individual = self.population[base_idx] + perturbation
            new_individual = np.clip(new_individual, -2.0, 2.0)
            new_population.append(new_individual)
        
        self.population = np.array(new_population)
        
        # Reset de parâmetros
        self.velocities = np.zeros_like(self.population)
        self.loudness = np.random.uniform(0.8, 1.2, self.population_size)
        self.pulse_rate = np.random.uniform(0.1, 0.3, self.population_size)
    
    def get_algorithm_stats(self) -> dict:
        """Retorna estatísticas úteis para monitoramento"""
        return {
            'iteration': self.iteration,
            'stagnation_counter': self.stagnation_counter,
            'diversity': self._calculate_diversity(),
            'avg_loudness': np.mean(self.loudness),
            'avg_pulse_rate': np.mean(self.pulse_rate),
            'current_fmax': self.fmax,
            'current_alpha': self.alpha
        }