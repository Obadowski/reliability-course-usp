import numpy as np
from scipy import stats

# ==========================================
# 1. FUNÇÃO PRINCIPAL DO TESTE K-S (ESTIMANDO OS MELHORES PARÂMETROS PELO MLE)
# ==========================================
def teste_aderencia_ks(dados, modelo_distribuicao='norm', nivel_confianca=0.95):
    """
    Aplica o teste K-S para verificar se os dados seguem a distribuição informada,
    considerando o nível de confiança escolhido pelo usuário.
    """
    dados = np.array(dados)
    
    # Passo 1: Ajusta os dados à distribuição para estimar os parâmetros (MLE)
    dist = getattr(stats, modelo_distribuicao)
    parametros = dist.fit(dados)
    
    # Passo 2: Executa o teste comparando a CDF empírica com a CDF teórica
    estatistica_d, p_valor = stats.kstest(dados, modelo_distribuicao, args=parametros)
    
    # Passo 3: Avaliação de Significância Dinâmica
    alpha = 1.0 - nivel_confianca
    aderencia = "Aceito (Segue o modelo)" if p_valor > alpha else "Rejeitado (Não segue)"
    
    return {
        "Estatística D": estatistica_d,
        "P-Valor": p_valor,
        "Resultado": aderencia,
        "Nível de Significância (Alpha)": alpha,
        "Parâmetros Ajustados": parametros
    }


# ==========================================
# 2. INTERFACE DO USUÁRIO (INPUTS)
# ==========================================
print("\n=== TESTE DE ADERÊNCIA KOLMOGOROV-SMIRNOV (K-S) ===")
try:
    # Recebe os dados de entrada
    entrada_dados = input("Digite os tempos de falha separados por espaço (Ex: 12.5 18.2 22.1): ").replace(',', '.')
    
    # Transforma o texto digitado numa lista de números (floats)
    tempos_user = [float(valor) for valor in entrada_dados.split()]
    
    # Pergunta qual modelo testar
    print("\nModelos disponíveis:")
    print("- 'norm'           (Normal/Gaussiana)")
    print("- 'expon'          (Exponencial)")
    print("- 'uniform'        (Uniforme)")
    print("- 'gamma'          (Gama)")
    print("- 'beta'           (Beta)")
    print("- 'weibull_min'    (Weibull Mínimo)")
    print("- 'weibull_max'    (Weibull Máximo)")
    print("- 'lognorm'        (Log-Normal)")
    print("- 'chi2'           (Qui-Quadrado)")
    print("- 't'              (t de Student)")
    print("- 'f'              (F de Snedecor)")
    print("- 'laplace'        (Laplace)")
    print("- 'cauchy'         (Cauchy)")
    print("- 'poisson'        (Poisson)")
    print("- 'binom'          (Binomial - Quantos sucessos em n tentativas?)")
    print("- 'nbinom'         (Binomial - Quantas tentativas até n sucessos?)")
    
    modelo_user = input("\nDigite a sigla do modelo que deseja testar: ").strip().lower()
    
    # --- NOVO: Recebe o Nível de Confiança ---
    entrada_confianca = input("\nDigite o nível de confiança desejado em porcentagem (Ex: 95 para 95%): ").replace(',', '.')
    confianca_pct = float(entrada_confianca)
    
    # Converte porcentagem para decimal (ex: 95 vira 0.95)
    if confianca_pct > 1:
        nivel_confianca_user = confianca_pct / 100.0
    else:
        # Se o usuário já digitar 0.95, o código entende e mantém
        nivel_confianca_user = confianca_pct
        
    print(f"\nRodando o teste K-S para o modelo '{modelo_user}' com {len(tempos_user)} dados...")
    
    # Chama a função passando a confiança
    resultado = teste_aderencia_ks(tempos_user, modelo_user, nivel_confianca_user)
    
    # Formata os parâmetros encontrados para não ficarem com muitas casas decimais
    parametros_formatados = tuple(round(p, 4) for p in resultado['Parâmetros Ajustados'])
    alpha_percentual = resultado['Nível de Significância (Alpha)'] * 100
    confianca_percentual = nivel_confianca_user * 100
    
    # Exibe os resultados finais
    print("-" * 50)
    print(f"Estatística D (Distância Máx): {resultado['Estatística D']:.4f}")
    print(f"P-Valor:                 {resultado['P-Valor']:.4f}")
    print(f"Nível de Significância:  {alpha_percentual:.1f}% (\u03B1 = {resultado['Nível de Significância (Alpha)']:.3f})")
    print(f"Decisão ({confianca_percentual:.1f}% Confiança): {resultado['Resultado']}")
    print(f"Parâmetros Ajustados:    {parametros_formatados}")
    print("-" * 50)

except AttributeError:
    print("\nErro: O modelo digitado não existe no SciPy. Verifique a sigla na lista acima.")
except ValueError:
    print("\nErro: Por favor, certifique-se de digitar apenas números válidos.")
except Exception as e:
    print(f"\nOcorreu um erro inesperado: {e}")