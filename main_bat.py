import numpy as np
from game.core import SurvivalGame, GameConfig
from game.agents import NeuralNetworkAgent
from bat_algorithm import BatAlgorithm
from test_trained_agent_bat import test_agent
import os
import sys
import time
import matplotlib.pyplot as plt


sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# Parâmetros otimizados
POPULATION_SIZE = 100
GENERATIONS = 1000
MAX_TIME = 12 * 3600  # 12 horas
WEIGHT_DIM = 611  # MUDANÇA: Nova arquitetura otimizada (28*16 + 17*8 + 9*3)


def game_fitness_function(population: np.ndarray) -> np.ndarray:
    """Função de fitness otimizada"""
    game_config = GameConfig(num_players=len(population), fps=120)
    agents = [NeuralNetworkAgent(weights) for weights in population]
    total_scores = np.zeros(len(agents))
    
    for i in range(5):
        game = SurvivalGame(config=game_config, render=False)
        
        while not game.all_players_dead():
            actions = []
            for idx, agent in enumerate(agents):
                if game.players[idx].alive:
                    state = game.get_state(idx, include_internals=True)
                    action = agent.predict(state)
                    actions.append(action)
                else:
                    actions.append(0)
            game.update(actions)
            
            if game.render:
                game.render_frame()
        
        for idx, player in enumerate(game.players):
            total_scores[idx] += player.score
    
    average_scores = total_scores / 5    
    return average_scores


def advanced_fitness_function(weights: np.ndarray) -> float:
    """Fitness avançada que considera sobrevivência + consistência"""
    scores = []
    survival_times = []
    
    for _ in range(5):
        game_config = GameConfig()
        game = SurvivalGame(config=game_config, render=False)
        agent = NeuralNetworkAgent(weights)
        
        frames_survived = 0
        while not game.all_players_dead():
            state = game.get_state(0, include_internals=True)
            action = agent.predict(state)
            game.update([action])
            frames_survived += 1
            
        scores.append(game.players[0].score) 
        survival_times.append(frames_survived)
    
    # Fitness combinado: score médio + bonus por consistência + bonus sobrevivência
    avg_score = np.mean(scores)
    consistency_bonus = 1.0 / (1.0 + np.std(scores))  # Menor desvio = maior bonus
    survival_bonus = np.mean(survival_times) * 0.01
    
    return avg_score + consistency_bonus + survival_bonus


def train_and_test():
    print("\n--- Iniciando Treinamento Otimizado com Bat Algorithm ---")
    print(f"Parâmetros: População={POPULATION_SIZE}, Gerações={GENERATIONS}, Pesos={WEIGHT_DIM}")
    
    # Usa arquitetura otimizada (611 pesos)
    ba = BatAlgorithm(
        population_size=POPULATION_SIZE,
        weight_dim=WEIGHT_DIM,  
        fmin=0,
        fmax=1,      
        alpha=0.95,  
        gamma=0.95   
    )
    
    best_weights_overall = None
    best_fitness_overall = -np.inf
    evolution_scores = []
    
    # Variáveis de monitoramento e early stopping
    best_fitness_history = []
    stagnation_counter = 0
    max_stagnation = 50
    
    start_time = time.time()
    
    for generation in range(GENERATIONS):
        # Verifica limite de tempo
        if (time.time() - start_time) > MAX_TIME:
            print(f"Interrompido por tempo (12h) na geração {generation}.")
            break
        
        start_generation = time.time()
        
        # Passa parâmetros de geração para o algoritmo adaptativo
        current_best_weights, current_best_fitness = ba.evolve(
            game_fitness_function, 
            parallel=True,
            generation=generation,
            max_generations=GENERATIONS
        )
        
        if current_best_fitness > best_fitness_overall:
            best_fitness_overall = current_best_fitness
            best_weights_overall = current_best_weights
            print(f'NOVO RECORDE! Geração {generation+1}: Fitness = {best_fitness_overall:.2f}')
            np.save("best_bat_weights.npy", best_weights_overall)
            stagnation_counter = 0  
        else:
            stagnation_counter += 1
        
        evolution_scores.append(best_fitness_overall)
        end = time.time()
        
        # Monitoramento detalhado a cada 10 gerações
        if generation % 10 == 0:
            stats = ba.get_algorithm_stats()
            print(f"Gen {generation + 1}/{GENERATIONS} | "
                  f"Atual: {current_best_fitness:.2f} | "
                  f"Melhor: {best_fitness_overall:.2f} | "
                  f"Diversidade: {stats['diversity']:.4f} | "
                  f"Estagnação: {stagnation_counter} | "
                  f"Tempo: {end-start_generation:.1f}s")
        else:
            print(f"Gen {generation + 1}/{GENERATIONS} | "
                  f"Fitness: {current_best_fitness:.2f} | "
                  f"Melhor: {best_fitness_overall:.2f} | "
                  f"({end-start_generation:.1f}s)")
        
        # Early stopping por estagnação
        if len(best_fitness_history) > 10:
            recent_improvement = (best_fitness_overall - 
                                np.mean(best_fitness_history[-10:]))
            
            if recent_improvement < 0.1:  
                stagnation_counter += 1
            
        if stagnation_counter >= max_stagnation:
            print(f"\nParada antecipada na geração {generation + 1}")
            print(f"Motivo: {max_stagnation} gerações sem melhoria significativa")
            break
            
        best_fitness_history.append(best_fitness_overall)
    
    total_time = time.time() - start_time
    print(f"\n--- Treinamento Concluído em {total_time/3600:.2f}h ---")
    print(f"Melhor Fitness Alcançado: {best_fitness_overall:.2f}")
    print(f"Total de Gerações: {len(evolution_scores)}")
    
    if best_weights_overall is not None:
        np.save("best_bat_weights.npy", best_weights_overall)
        print("Melhores pesos salvos em 'best_bat_weights.npy'")
        
        plot_evolution(evolution_scores)
        
        # Teste rápido com 3 jogos primeiro
        print("\n--- Teste Rápido (3 jogos) ---")
        quick_test_scores = []
        for i in range(3):
            game_config = GameConfig(num_players=1)
            game = SurvivalGame(config=game_config, render=False)
            agent = NeuralNetworkAgent(best_weights_overall)
            
            while not game.all_players_dead():
                state = game.get_state(0, include_internals=True)
                action = agent.predict(state)
                game.update([action])
                
            quick_test_scores.append(game.players[0].score)
            print(f"Teste rápido {i+1}: {game.players[0].score:.2f}")
        
        print(f"Score médio teste rápido: {np.mean(quick_test_scores):.2f} ± {np.std(quick_test_scores):.2f}")
        
        # Teste completo com visualização
        print("\n--- Iniciando Teste Completo (30 jogos) ---")
        test_agent(best_weights_overall, num_tests=30, render=False)
    else:
        print("Nenhum peso ótimo encontrado.")


def plot_evolution(scores):
    """Plota gráfico melhorado da evolução do agente"""
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(range(1, len(scores)+1), scores, 'b-', linewidth=2, label='Melhor Fitness')
    
    if len(scores) > 10:
        z = np.polyfit(range(1, len(scores)+1), scores, 1)
        p = np.poly1d(z)
        plt.plot(range(1, len(scores)+1), p(range(1, len(scores)+1)), 
                'r--', alpha=0.7, label='Tendência')
    
    plt.xlabel("Geração")
    plt.ylabel("Melhor Fitness")
    plt.title("Evolução do Agente - Algoritmo dos Morcegos (Bat Algorithm)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    if len(scores) > 1:
        improvements = [scores[i] - scores[i-1] for i in range(1, len(scores))]
        plt.plot(range(2, len(scores)+1), improvements, 'g-', linewidth=1, alpha=0.7)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.xlabel("Geração")
        plt.ylabel("Melhoria do Fitness")
        plt.title("Melhoria por Geração")
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("bat_evolution.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Gráfico salvo como 'bat_evolution.png'")


def print_training_summary():
    """Imprime resumo das configurações de treinamento"""
    print("=" * 60)
    print("CONFIGURAÇÃO DE TREINAMENTO OTIMIZADA")
    print("=" * 60)
    print(f"População: {POPULATION_SIZE} morcegos")
    print(f"Gerações máximas: {GENERATIONS}")
    print(f"Arquitetura da rede: 27+1 → 16 → 8 → 3 ({WEIGHT_DIM} pesos)")
    print(f"Jogos por avaliação: 5 (era 3)")
    print(f"FPS do jogo: 120 (otimizado)")
    print(f"Tempo limite: {MAX_TIME/3600:.0f} horas")
    print(f"Parada antecipada: 50 gerações sem melhoria")
    print("=" * 60)


if __name__ == "__main__":
    print_training_summary()
    train_and_test()