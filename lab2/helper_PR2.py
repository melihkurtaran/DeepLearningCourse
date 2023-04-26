#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 09:10:15 2022

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Función que realiza la gráfica
def my_plot(w1, w2, bias, test_inputs, correct_outputs):
    """
    Esta función recibe como entrada los parámetros de una recta w1, w2 y
    bias. También recibe un conjunto de puntos en 'test_inputs' formado por
    un array de tuplas, y un conjunto de etiquetas en 'test_outputs'
    formado por un array de elementos booleanos.
    La salida es una gráfica en la que aparecen todos los puntos. Los
    puntos aparecen azules si su etiqueta es True y rojos si su etiqueta
    es False.
    También aparece la recta (si está en el rango donde están los puntos).
    La recta aparece por duplicado, una azul y una roja. Esto es para
    diferenciar las zonas que separa la recta. La zona azul es la zona en
    la que los puntos responden a la ecuación w1·x1 + w2·x2 + bias > 0
    """
    x_red = np.array([])
    y_red = np.array([])
    x_blue = np.array([])
    y_blue = np.array([])

    for k in range(len(correct_outputs)):
        if not correct_outputs[k]:
            x_red = np.append(x_red, test_inputs[k][0])
            y_red = np.append(y_red, test_inputs[k][1])
        else:
            x_blue = np.append(x_blue, test_inputs[k][0])
            y_blue = np.append(y_blue, test_inputs[k][1])

    # Mira si la zona positiva está por encima o por debajo de la recta en
    # x=0. Si w2>0 la zona positiva está arriba; si no, está abajo.
    w2 = w2 if w2 != 0 else 0.001
    z = np.sign(w2)

    # Mira si la pendiente de la recta es positiva
    m = np.sign(-w1/w2)
    m = m if m != 0 else 1

    # Calcula los rangos
    x_min = min(np.append(x_red, x_blue))
    x_max = max(np.append(x_red, x_blue))
    y_min = min(np.append(y_red, y_blue))
    y_max = max(np.append(y_red, y_blue))
    rango_x = x_max - x_min
    rango_y = y_max - y_min

    # Calcula valores de los puntos que definen las rectas
    eps = 0.02 * max([rango_x, rango_y])
    x_val1 = np.array([x_min - 0.25 * rango_x, x_max + 0.25 * rango_x])
    x_val2 = x_val1 + eps * z * m
    y_val1 = -w1 / w2 * x_val1 - bias / w2
    y_val2 = y_val1 - eps * z

    # Formatos de puntos y líneas
    fmt_red = 'or'
    fmt_blue = 'ob'
    fmt_val1 = '-b'
    fmt_val2 = '-r'

    # Dibuja gráfica con puntos y rectas
    plt.plot(x_red, y_red, fmt_red, linewidth=1, markersize=10)
    plt.plot(x_blue, y_blue, fmt_blue, linewidth=1, markersize=10)
    plt.plot(x_val1, y_val1, fmt_val1, linewidth=2, markersize=1)
    plt.plot(x_val2, y_val2, fmt_val2, linewidth=2, markersize=1)
    plt.ylim(y_min - 0.25 * rango_y, y_max + 0.25 * rango_y)
    plt.show()


def evaluate(weight1, weight2, bias, test_inputs, correct_outputs,
             extended=True):

    outputs = []

    # Genera las salidas
    for test_input, correct_output in zip(test_inputs, correct_outputs):
        # Calcula la combinación lineal de cada punto con los pesos
        linear_combination = weight1 * test_input[0] \
            + weight2 * test_input[1] + bias
        # La predicción es True si la combinación lineal es >= que cero
        output = int(linear_combination >= 0)
        # Indica si la predicción es correcta o no
        is_correct_string = 'Yes' if output == correct_output else 'No'
        # Añade el elemento actual a la lista de salida
        outputs.append([test_input[0], test_input[1], linear_combination,
                        output, is_correct_string])

    # Calcula datos de salida
    num_wrong = len([output[4] for output in outputs if output[4] == 'No'])
    output_frame = pd.DataFrame(outputs, columns=['Input 1', '  Input 2',
                                                  '  Linear Combination',
                                                  '  Activation Output',
                                                  '  Is Correct'])
    if not num_wrong:
        print('Perfect!\n')
    else:
        print('You got {} wrong.\n'.format(num_wrong))

    # Datos extendidos
    if extended:
        print(output_frame.to_string(index=False))

    my_plot(weight1, weight2, bias, test_inputs, correct_outputs)


# Función para visualizar el resultado

def plot_dots(blue_points_calc, red_points_calc,
              blue_points_gt=[], red_points_gt=[]):

    x_blue_c = np.array([])
    y_blue_c = np.array([])
    x_red_c = np.array([])
    y_red_c = np.array([])

    x_blue_gt = np.array([])
    y_blue_gt = np.array([])
    x_red_gt = np.array([])
    y_red_gt = np.array([])
    
    for (x, y) in blue_points_calc:
        x_blue_c = np.append(x_blue_c, x)
        y_blue_c = np.append(y_blue_c, y)

    for (x, y) in red_points_calc:
        x_red_c = np.append(x_red_c, x)
        y_red_c = np.append(y_red_c, y)

    for (x, y) in blue_points_gt:
        x_blue_gt = np.append(x_blue_gt, x)
        y_blue_gt = np.append(y_blue_gt, y)

    for (x, y) in red_points_gt:
        x_red_gt = np.append(x_red_gt, x)
        y_red_gt = np.append(y_red_gt, y)

    fmt_blue = 'ob'
    fmt_red = 'or'

    # Dibuja gráfica con puntos y rectas
    plt.plot(x_blue_c, y_blue_c, fmt_blue, linewidth=1, markersize=6)
    plt.plot(x_red_c, y_red_c, fmt_red, linewidth=1, markersize=6)
    plt.plot(x_blue_gt, y_blue_gt, fmt_blue, linewidth=1, markersize=10)
    plt.plot(x_red_gt, y_red_gt, fmt_red, linewidth=1, markersize=10)
    plt.ylim(0.3, 1.2)
    plt.xlim(0.3, 1.2)
    plt.show()
