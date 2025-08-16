import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, wilcoxon
from game.core import SurvivalGame, GameConfig
from game.agents import NeuralNetworkAgent
import warnings
warnings.filterwarnings('ignore')

def test_agent(weights: np.ndarray, num_tests: int = 30, render: bool = True):
    """Testa o agente neural treinado com Bat Algorithm"""
    print(f"\n--- Testando Agente Neural (Bat Algorithm) por {num_tests} vezes ---")
    
    game_config = GameConfig(render_grid=True)
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
    
    avg_score = np.mean(total_scores)
    std_score = np.std(total_scores)
    
    print(total_scores)
    print(f"\nResultados Finais após {num_tests} testes:")
    print(f"Score Médio: {avg_score:.2f}")
    print(f"Desvio Padrão do Score: {std_score:.2f}")
    print(f"Média - DP: {avg_score - std_score:.2f}")

    return total_scores, avg_score, std_score

def run_comparative_analysis(bat_scores):
    """Executa análise comparativa completa entre todos os agentes"""
    
    rule_based_result = [12.69, 16.65, 6.97, 2.79, 15.94, 10.22, 21.90, 4.35, 6.22,
                         9.95, 19.94, 20.56, 15.74, 17.68, 7.16, 15.68, 2.37, 15.43, 
                         15.13, 22.50, 25.82, 15.85, 17.02, 16.74, 14.69, 11.73, 13.80, 
                         15.13, 12.35, 16.19]

    neural_agent_result = [38.32, 54.53, 61.16, 27.55, 16.08, 26.00, 25.33, 18.30,
                           39.76, 48.17, 44.77, 47.54, 75.43, 23.68, 16.83, 15.81, 
                           67.17, 53.54, 33.59, 49.24, 52.65, 16.35, 44.05, 56.59, 
                           63.23, 43.96, 43.82, 19.19, 28.36, 18.65]

    human_result = [27.34, 17.63, 39.33, 17.44, 1.16, 24.04, 29.21, 18.92, 25.71, 
                    20.05, 31.88, 15.39, 22.50, 19.27, 26.33, 23.67, 16.82, 28.45,
                    12.59, 33.01, 21.74, 14.23, 27.90, 24.80, 11.35, 30.12, 17.08, 
                    22.96, 9.41, 35.22]
    
    agents = {
        'Agente Bat': bat_scores,
        'Agente Baseado em Regras': rule_based_result,
        'Agente Neural': neural_agent_result,
        'Agente Humano': human_result
    }
    
    print("\n" + "="*80)
    print("ANÁLISE COMPARATIVA DOS AGENTES")
    print("="*80)
    
    create_results_table(agents)
    
    perform_statistical_tests(agents)
    
    create_boxplot(agents)

def create_results_table(agents):
    """Cria tabela com todos os resultados"""
    table_data = []
    
    for agent_name, scores in agents.items():
        row = scores + [np.mean(scores), np.std(scores)]
        table_data.append(row)
    
    columns = [f'Teste {i+1}' for i in range(30)] + ['Média', 'Desvio Padrão']
    df = pd.DataFrame(table_data, index=list(agents.keys()), columns=columns)
    
    print("\nTABELA DE RESULTADOS")
    print("-" * 80)
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    print(df.round(2))
    
    return df

def perform_statistical_tests(agents):
    """Realiza testes estatísticos entre todos os pares de agentes"""
    agent_names = list(agents.keys())
    results = []
    
    print("\n" + "="*80)
    print("TESTES ESTATÍSTICOS")
    print("="*80)
    
    for i in range(len(agent_names)):
        for j in range(i+1, len(agent_names)):
            agent1 = agent_names[i]
            agent2 = agent_names[j]
            scores1 = agents[agent1]
            scores2 = agents[agent2]
            
            # Teste t
            t_stat, t_pvalue = ttest_ind(scores1, scores2)
            
            # Teste de Wilcoxon
            differences = np.array(scores1) - np.array(scores2)
            w_stat, w_pvalue = wilcoxon(differences)
            
            # Verificar significância (α = 0.05)
            t_significant = "SIM" if t_pvalue < 0.05 else "NÃO"
            w_significant = "SIM" if w_pvalue < 0.05 else "NÃO"
            
            print(f"\n{agent1} vs {agent2}")
            print("-" * 60)
            print(f"Teste t independente:")
            print(f"  Estatística t: {t_stat:.4f}")
            print(f"  p-value: {t_pvalue:.6f}")
            print(f"  Diferença significativa (α=0.05): {t_significant}")
            
            print(f"Teste de Wilcoxon (signed-rank):")
            print(f"  Estatística: {w_stat:.4f}")
            print(f"  p-value: {w_pvalue:.6f}")
            print(f"  Diferença significativa (α=0.05): {w_significant}")
            
            results.append({
                'Comparação': f"{agent1} vs {agent2}",
                'Teste t - p-value': t_pvalue,
                'Teste t - Significativo': t_significant,
                'Wilcoxon - p-value': w_pvalue,
                'Wilcoxon - Significativo': w_significant
            })
    
    test_df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("RESUMO DOS TESTES ESTATÍSTICOS")
    print("="*80)
    print(test_df.round(6))
  
    return results

def create_boxplot(agents):
    """Cria boxplot comparativo dos resultados"""
    data_for_plot = []
    labels = []
    
    for agent_name, scores in agents.items():
        data_for_plot.extend(scores)
        labels.extend([agent_name] * len(scores))
    
    df_plot = pd.DataFrame({
        'Agente': labels,
        'Score': data_for_plot
    })
    
    plt.figure(figsize=(12, 8))
    
    # Boxplot com seaborn
    box_plot = sns.boxplot(data=df_plot, x='Agente', y='Score', palette='Set2')
    
    plt.title('Comparação de Desempenho dos Agentes\n(Boxplot dos Scores)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Agente', fontsize=12, fontweight='bold')
    plt.ylabel('Score', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Adicionar média como ponto
    means = [np.mean(scores) for scores in agents.values()]
    plt.scatter(range(len(means)), means, color='red', s=100, marker='D', 
                label='Média', zorder=3)
    
    plt.legend()
    plt.tight_layout()
    
    plt.savefig('agentes_boxplot_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Boxplot salvo em: agentes_boxplot_comparison.png")

if __name__ == "__main__":
    if os.path.exists('best_bat_weights.npy'):
        best_trained_weights = np.load('best_bat_weights.npy')
        
        # Testar o agente
        bat_scores, avg_score, std_score = test_agent(best_trained_weights, num_tests=30, render=False)
        
        # Executar análise comparativa
        run_comparative_analysis(bat_scores)
            
    else:
        print("Arquivo 'best_bat_weights.npy' não encontrado. Execute main_bat.py primeiro.")