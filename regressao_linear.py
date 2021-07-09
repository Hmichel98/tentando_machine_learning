import numpy as np
from typing import Union
"""
Exemplo:
    x = np.array([ 2,  4,  8, 16, 32])
    y = np.array([ 4,  8, 16, 31, 64])
    predict = linear_regression(x, y)

    Para retornar os coeficientes
    
    coeficiente_angular = predict.__closure__[0].cell_contents
    coeficiente_linear = predict.__closure__[1].cell_contents
"""


def regressao_linear(x: np.ndarray, y: np.ndarray) -> type(lambda x: m * x + n):
    """ 
    Implementação da regressão linear simples. Utiliza:
    - Closure: isto é, retorna uma função interna que utiliza variáveis do enclosing scope;
    - Gradient Descent: algorítimo iterativo para achar valores locais mínimos;
            - novo_x = x_velho - gamma * derivada_da_funcao(x_velho).
    - Mean Squared Error: distância entre o verdadeiro output e o output do modelo.
    """
    # Também chamado de epoch
    n_iteracoes = 1000
    # Quão rápido o valor é atualizado no gradient descent
    gamma = 0.001

    # É necessário para a média do Mean Squared Error
    length = len(x) 

    # Coeficiente angular (m)
    m = 0
    # Coeficiente linear (n)
    n = 0

    for _ in range(n_iteracoes):
        # Fator comum. Output do modelo
        y_model = m * x + n

        # Derivada parcial em função do coeficiente angular
        dl_dm = (-2/length) * np.sum(x * (y - y_model)) 
        # Derivada parcial em função do coeficiente linear
        dl_dn = (-2/length) * np.sum(y - y_model) 

        # Atualização dos coeficientes utilizando gradient descent 
        m -= gamma * dl_dm
        n -= gamma * dl_dn

    def predict(new_x: Union[float, int]) -> float:
        """Retorna a previsão do modelo linear em um dado x"""
        return m * new_x + n

    # Será retornado uma função que poderá ser usada para prever novos valores
    # isto é, interpolar ou extrapolar
    return predict



