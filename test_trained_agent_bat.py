import numpy as np
from game.core import SurvivalGame, GameConfig
from game.agents import NeuralNetworkAgent
import os

def test_agent(weights: np.ndarray, num_tests: int = 30, render: bool = False):
    """Testa o agente treinado por um número especificado de vezes"""
    print(f"\n--- Testando Agente Neural Treinado por {num_tests} vezes ---")
    
    game_config = GameConfig(render_grid=False)
    total_scores = []
    
    for i in range(num_tests):
        game = SurvivalGame(config=game_config, render=render)
        agent = NeuralNetworkAgent(weights)
        
        while not game.all_players_dead():
            state = game.get_state(0, include_internals=True)
            action = agent.predict(state)
            game.update([action])
            
            if render:
                game.render_frame()
        
        final_score = game.players[0].score
        total_scores.append(final_score)
        print(f"Teste {i+1}/{num_tests}: Score Final = {final_score:.2f}")
    
    # Estatísticas finais
    avg_score = np.mean(total_scores)
    std_score = np.std(total_scores)
    
    print(f"\nResultados Finais após {num_tests} testes:")
    print(f"Score Médio: {avg_score:.2f}")
    print(f"Desvio Padrão: {std_score:.2f}")
    print(f"Score Mínimo: {np.min(total_scores):.2f}")
    print(f"Score Máximo: {np.max(total_scores):.2f}")
    
    return total_scores, avg_score, std_score

def compare_agents():
    """Compara diferentes agentes (para análise estatística)"""
    print("\n--- Comparação de Agentes ---")
    
    # Carrega pesos treinados
    if os.path.exists('best_bat_weights.npy'):
        bat_weights = np.load('best_bat_weights.npy')
        print("Testando Agente Bat Algorithm...")
        bat_scores, bat_avg, bat_std = test_agent(bat_weights, num_tests=30, render=False)
    
    # Teste com agente baseado em regras (se disponível)
    if os.path.exists('best_weights.npy'):
        from game.agents import RuleBasedAgent
        rule_weights = np.load('best_weights.npy')
        print("\nTestando Agente Baseado em Regras...")
        
        rule_scores = []
        for i in range(30):
            game_config = GameConfig()
            game = SurvivalGame(config=game_config, render=False)
            agent = RuleBasedAgent(
                config=game_config,
                danger_threshold=rule_weights[0],
                lookahead_cells=rule_weights[1],
                diff_to_center_to_move=rule_weights[2]
            )
            
            while not game.all_players_dead():
                state = game.get_state(0, include_internals=True)
                action = agent.predict(state)
                game.update([action])
            
            rule_scores.append(game.players[0].score)
        
        rule_avg = np.mean(rule_scores)
        rule_std = np.std(rule_scores)
        
        print(f"Regras - Média: {rule_avg:.2f}, Desvio: {rule_std:.2f}")
        
        # Teste estatístico
        from scipy.stats import ttest_ind, wilcoxon
        
        if 'bat_scores' in locals():
            t_stat, t_p = ttest_ind(bat_scores, rule_scores)
            print(f"\nTeste t: t={t_stat:.4f}, p={t_p:.4f}")
            
            try:
                w_stat, w_p = wilcoxon(bat_scores, rule_scores)
                print(f"Wilcoxon: W={w_stat:.4f}, p={w_p:.4f}")
            except:
                print("Não foi possível executar o teste de Wilcoxon")

if __name__ == "__main__":
    if os.path.exists('best_bat_weights.npy'):
        best_weights = np.load('best_bat_weights.npy')
        test_agent(best_weights, num_tests=30, render=True)
        compare_agents()
    else:
        print("Arquivo 'best_bat_weights.npy' não encontrado. Execute main_bat.py primeiro.")
