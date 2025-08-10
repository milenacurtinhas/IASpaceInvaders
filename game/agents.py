import numpy as np
from abc import ABC, abstractmethod
from typing import List

class Agent(ABC):
    """Interface para todos os agentes."""
    @abstractmethod
    def predict(self, state: np.ndarray) -> int:
        """Faz uma previsão de ação com base no estado atual."""
        pass

class HumanAgent(Agent):
    """Agente controlado por um humano (para modo manual)"""
    def predict(self, state: np.ndarray) -> int:
        # O estado é ignorado - entrada vem do teclado
        return 0  # Padrão: não fazer nada (será sobrescrito pela entrada do usuário no manual_play.py)

class NeuralNetworkAgent(Agent):
    """Agente baseado em rede neural otimizada para o jogo de sobrevivência"""
    
    def __init__(self, weights: np.ndarray = None):
        """
        Inicializa o agente da rede neural.
        
        Args:
            weights: Array com os pesos da rede neural (611 valores)
                    Se None, inicializa com pesos aleatórios usando Xavier
        """
        # Arquitetura otimizada da rede neural
        self.layer_sizes = [
            (28, 16),  # Entrada: 27 sensores + 1 bias → 16 neurônios
            (17, 8),   # Oculta: 16 + 1 bias → 8 neurônios  
            (9, 3)     # Saída: 8 + 1 bias → 3 ações
        ]
        
        # Total de pesos: 28*16 + 17*8 + 9*3 = 448 + 136 + 27 = 611
        self.total_weights = self._calculate_total_weights()
        
        if weights is not None:
            if len(weights) != self.total_weights:
                raise ValueError(f"Esperado {self.total_weights} pesos, recebido {len(weights)}")
            self.weights = weights.copy()
        else:
            # Inicialização Xavier para redes com tanh
            self.weights = self._initialize_xavier()
        
        # Converte os pesos em matrizes para cada camada
        self.layers = self._weights_to_layers()
    
    def _calculate_total_weights(self) -> int:
        """Calcula o número total de pesos necessários"""
        return sum(input_size * output_size for input_size, output_size in self.layer_sizes)
    
    def _initialize_xavier(self) -> np.ndarray:
        """Inicialização Xavier/Glorot para ativação tanh"""
        weights = []
        
        for input_size, output_size in self.layer_sizes:
            # Limite Xavier: sqrt(6 / (fan_in + fan_out))
            limit = np.sqrt(6.0 / (input_size + output_size))
            layer_weights = np.random.uniform(-limit, limit, input_size * output_size)
            weights.extend(layer_weights)
        
        return np.array(weights)
    
    def _weights_to_layers(self) -> List[np.ndarray]:
        """Converte o array de pesos em matrizes de camadas"""
        layers = []
        start_idx = 0
        
        for input_size, output_size in self.layer_sizes:
            end_idx = start_idx + (input_size * output_size)
            layer_weights = self.weights[start_idx:end_idx]
            # Reshape para matriz (input_size, output_size)
            layer_matrix = layer_weights.reshape(input_size, output_size)
            layers.append(layer_matrix)
            start_idx = end_idx
        
        return layers
    
    def _tanh(self, x: np.ndarray) -> np.ndarray:
        """Função de ativação tanh com clipping para estabilidade"""
        x_clipped = np.clip(x, -500, 500)  # Evita overflow
        return np.tanh(x_clipped)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Função softmax para camada de saída"""
        x_shifted = x - np.max(x)  # Estabilidade numérica
        exp_values = np.exp(x_shifted)
        return exp_values / np.sum(exp_values)
    
    def predict(self, state: np.ndarray) -> int:
        """
        Faz uma previsão de ação com base no estado atual.
        
        Args:
            state: Estado do jogo (27 valores dos sensores)
            
        Returns:
            int: Ação escolhida (0=parado, 1=cima, 2=baixo)
        """
        if len(state) != 27:
            raise ValueError(f"Estado deve ter 27 valores, recebido {len(state)}")
        
        # Forward pass pela rede neural
        
        # Camada 1: Entrada → Oculta 1
        # Adiciona bias (1.0) ao estado de entrada
        x = np.append(state, 1.0)  # 28 valores
        z1 = self._tanh(np.dot(x, self.layers[0]))  # 16 valores
        
        # Camada 2: Oculta 1 → Oculta 2  
        # Adiciona bias à primeira camada oculta
        z1_with_bias = np.append(z1, 1.0)  # 17 valores
        z2 = self._tanh(np.dot(z1_with_bias, self.layers[1]))  # 8 valores
        
        # Camada 3: Oculta 2 → Saída
        # Adiciona bias à segunda camada oculta
        z2_with_bias = np.append(z2, 1.0)  # 9 valores
        output = self._softmax(np.dot(z2_with_bias, self.layers[2]))  # 3 valores
        
        # Retorna a ação com maior probabilidade
        return np.argmax(output)
    
    def get_action_probabilities(self, state: np.ndarray) -> np.ndarray:
        """
        Retorna as probabilidades de cada ação em vez da ação escolhida.
        Útil para análise e debugging.
        
        Args:
            state: Estado do jogo (27 valores)
            
        Returns:
            np.ndarray: Probabilidades das 3 ações [parado, cima, baixo]
        """
        if len(state) != 27:
            raise ValueError(f"Estado deve ter 27 valores, recebido {len(state)}")
        
        # Forward pass (mesmo código do predict)
        x = np.append(state, 1.0)
        z1 = self._tanh(np.dot(x, self.layers[0]))
        z1_with_bias = np.append(z1, 1.0)
        z2 = self._tanh(np.dot(z1_with_bias, self.layers[1]))
        z2_with_bias = np.append(z2, 1.0)
        output = self._softmax(np.dot(z2_with_bias, self.layers[2]))
        
        return output
    
    def update_weights(self, new_weights: np.ndarray):
        """
        Atualiza os pesos da rede neural.
        Útil durante o treinamento com algoritmo dos morcegos.
        
        Args:
            new_weights: Novos pesos (611 valores)
        """
        if len(new_weights) != self.total_weights:
            raise ValueError(f"Esperado {self.total_weights} pesos, recebido {len(new_weights)}")
        
        self.weights = new_weights.copy()
        self.layers = self._weights_to_layers()
    
    def get_weights(self) -> np.ndarray:
        """Retorna uma cópia dos pesos atuais"""
        return self.weights.copy()
    
    def get_network_info(self) -> dict:
        """
        Retorna informações sobre a arquitetura da rede.
        Útil para debugging e logging.
        """
        return {
            'layer_sizes': self.layer_sizes,
            'total_weights': self.total_weights,
            'weights_per_layer': [input_size * output_size 
                                for input_size, output_size in self.layer_sizes],
            'architecture': '27+1 → 16 → 8 → 3'
        }

class RuleBasedAgent(Agent):
    """
    Agente baseado em regras para comparação.
    Mantido do código original do algoritmo genético.
    """
    
    def __init__(self, danger_threshold: float = 5.0, 
                 lookahead_cells: int = 3, 
                 diff_to_center_to_move: float = 2.0):
        """
        Args:
            danger_threshold: Distância mínima para considerar obstáculo perigoso
            lookahead_cells: Quantas células à frente verificar
            diff_to_center_to_move: Diferença do centro necessária para se mover
        """
        self.danger_threshold = danger_threshold
        self.lookahead_cells = int(lookahead_cells)
        self.diff_to_center_to_move = diff_to_center_to_move
    
    def predict(self, state: np.ndarray) -> int:
        """
        Lógica baseada em regras para decidir ação.
        
        Returns:
            0: Ficar parado
            1: Mover para cima  
            2: Mover para baixo
        """
        # Implementação simplificada - pode ser expandida conforme necessário
        # Esta é uma versão básica para manter compatibilidade
        
        # Analisa os sensores do estado (assumindo que os primeiros valores são posições)
        if len(state) < 3:
            return 0
        
        # Lógica simples: se há obstáculo próximo, move
        danger_detected = np.any(state[:self.lookahead_cells] < self.danger_threshold)
        
        if danger_detected:
            # Move baseado na posição relativa ao centro
            player_y_relative = state[0] if len(state) > 0 else 0
            
            if player_y_relative > self.diff_to_center_to_move:
                return 2  # Mover para baixo
            elif player_y_relative < -self.diff_to_center_to_move:
                return 1  # Mover para cima
            else:
                return 1 if np.random.rand() > 0.5 else 2  # Movimento aleatório
        
        return 0  # Ficar parado se não há perigo