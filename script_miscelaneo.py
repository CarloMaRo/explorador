import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt
import matplotlib.dates  as mdates
import seaborn           as sns

import re

import statsmodels.tsa.stattools     as sts                    # Este también sirve para eltest de estacionariedad de Dickey-ful
import statsmodels.graphics.tsaplots as sgt
import statsmodels.api               as sm

from sklearn.metrics import mean_absolute_error, mean_squared_error

from statsmodels.tsa.api                import VAR                # Esto es para series multivariadas
from statsmodels.tsa.seasonal           import seasonal_decompose
from statsmodels.tsa.stattools          import grangercausalitytests  # Este sirve para crear la MATRIZ de causalidad de granger
from scipy.stats.distributions          import chi2

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score

import cv2
'''
try:
  from pyspark.sql.functions import col
  from pyspark.sql.functions import count
  from pyspark.sql.functions import expr
  from pyspark.sql.functions import sum
  from pyspark.sql.functions import desc
  from pyspark.sql.functions import corr
  from pyspark.sql.functions import when
  from pyspark.sql.functions import lit
  from pyspark.sql.functions import to_date
  from pyspark.sql.functions import regexp_replace
except:
  print('no se importa nada de pyspark')   
'''


######################################################################################################################
################################# FUNCIÓN PARA GENERAR UNA CAMINATA ALEATORIA ########################################
######################################################################################################################


def generador_caminata_aleatoria_con_atipicos(valor_inicial_caminata, cant_pasos_a_generar, porcentaje_cant_atipicos = 0.0, factor_mult_atipicos = 10, semilla = 42):
  assert (porcentaje_cant_atipicos >= 0.0) & (porcentaje_cant_atipicos <= 1)          # Con "assert", python mandará error si la condición no se cumple, si la cnodición se cumple, el código sigue normal, como si nada

  np.random.seed(semilla)                                                             # Fijamos la semilla para la reproducibilidad del código
  valores_posibles_para_pasos = [-1,1]                                                # Valores posibles de la caminata aleatoria

  pasos_aleatorios   = np.random.choice(a = valores_posibles_para_pasos, size = cant_pasos_a_generar-1)  # Se crea un arreglo con los pasos aleatorios que tendrá la caminata
  caminata_aleatoria = np.append(valor_inicial_caminata, pasos_aleatorios).cumsum(0)                     # Se crea la caminata tomando el valor inicial y de ahí en adelante, sumándole los pasos aleatorios uno tras otro (con el acumulado)
  #print("Estos son los pasos aleatorios de la caminata\n",pasos_aleatorios,"\n")
  #print("Se imprime la caminata aleatoria (SIN atípicos)\n",caminata_aleatoria, "\n")

  nro_decimales_a_tomar       = 0
  cant_atipicos               = int(np.round(porcentaje_cant_atipicos * cant_pasos_a_generar,nro_decimales_a_tomar)) # Generamos una cantidad de atípicos según el procentaje fijado
  indices                     = np.random.randint(0, len(caminata_aleatoria), cant_atipicos)                         # Generamos posicisiones (aleatorias también) de la caminata aleatoria, donde tendremos atípios

  caminata_aleatoria[indices] = caminata_aleatoria[indices] + pasos_aleatorios[indices + 1] * factor_mult_atipicos   # En las posiciones de la caminata antes fijadas, creamos datos atípicos multiplicados por 10
  #print("Se imprime la caminata aleatoria (CON atípicos)\n",caminata_aleatoria, "\n")

  if porcentaje_cant_atipicos == 0:
    return caminata_aleatoria
  else:
    return caminata_aleatoria, indices


######################################################################################################################
############################################# FUNCIONES PARA EXPLORACIÓN #############################################
######################################################################################################################


def impresor_de_caracteres(elemento, nro_max_de_caracteres, espacio_antes = True):
  elemento_string     = len(str(elemento))
  elemento_a_imprimir = elemento
  for i in range(nro_max_de_caracteres - elemento_string):
    elemento_a_imprimir = ' ' + str(elemento_a_imprimir) if espacio_antes == True else str(elemento_a_imprimir) + ' '
  return elemento_a_imprimir


def max_cantidad_de_caractares_en_listado(listado):
  for cont, i in enumerate(listado):
    long_i = len(str(i))
    if cont == 0:
      max_caracteres = long_i
    else:
      if max_caracteres < long_i:
        max_caracteres = long_i
  return max_caracteres


def completar_decimal(string, nro_decimales):
    for i in range(len(string), nro_decimales):
        string = string + '0'
    return string


def configurar_nro_decimal(val, nro_decimales, porcentaje = False):
    multiplicador    = 100 if porcentaje == True else 1
    val              = np.round(val*multiplicador, nro_decimales)
    val_str          = str(val)
    ent_str          = val_str.split('.')[0]
    dec_str          = '0' if val == int(val) else val_str.split('.')[1]
    dec_str_completo = completar_decimal(dec_str, nro_decimales)
    return ent_str + '.' + dec_str_completo


def acumulador_porcentaje(lista):
    valor      = 1
    lista_acum = []
    for elemento_i in lista:
        lista_acum.append(valor)
        valor = valor - elemento_i
    return lista_acum


def tipos_columnas(dataframe):
  arr_numerico         = []
  arr_categorico       = []
  arr_booleano         = []
  arr_fecha            = []
  arr_delta_fecha      = []
  arr_dato_desconocido = []
  for col_i in dataframe.columns.tolist():
    tipo_dato = dataframe[col_i].dtype
    if   ( tipo_dato == 'int64'  ) | ( tipo_dato == 'float64' ):
      arr_numerico.append(col_i)
    elif ( tipo_dato == 'object' ) | ( tipo_dato == 'category'):
      arr_categorico.append(col_i)
    elif ( tipo_dato == 'bool'):
      arr_booleano.append(col_i)
    elif ( tipo_dato == 'datetime64[ns]' ):
      arr_fecha.append(col_i)
    elif ( tipo_dato == 'timedelta[ns]'):
      arr_delta_fecha.append(col_i)
  return arr_numerico, arr_categorico, arr_booleano, arr_fecha, arr_delta_fecha, arr_dato_desconocido


# -------------------------------------------------------------------------
# --------------- PARA DETERMINAR LA CARDINALIDAD DE LA DATA --------------
# -------------------------------------------------------------------------

def cardinalidad(df_car):

  encabezados = df_car.columns.values.tolist()
  nro_filas = len(df_car)
  nro_cols  = len(df_car.columns.tolist())

  nro_datos_repetidos = df_car.duplicated().sum()

  print('NRO DE COLUMNAS        = {}'.format(nro_cols))
  print('NRO DE FILAS           = {}'.format(nro_filas))
  print('NRO DE FILAS REPETIDAS = {}'.format(nro_datos_repetidos))

  tipos_de_datos = []
  for i in encabezados:
    tipos_de_datos.append(df_car[i].dtype)
    #tipos_de_datos.append(df_car[i].dtypes)

  max_cifras_cat          = 6
  max_cifras_nan          = 6
  max_cifras_tipo_de_dato = max_cantidad_de_caractares_en_listado(listado = tipos_de_datos)
  max_cifras_variable     = max_cantidad_de_caractares_en_listado(listado = encabezados)
  max_cifras_contador     = len(str(len(encabezados)))

  for cont, i in enumerate(encabezados):
  
    nro_categ_i    = len(df_car[i].unique())
    #nro_categ_i    = len(pd.unique(df_car[i]))
    categ_impr     = impresor_de_caracteres(elemento = nro_categ_i, nro_max_de_caracteres = max_cifras_cat)   
    
    nro_nans_i     = df_car.isna().sum()[cont]
    nans_impr      = impresor_de_caracteres(elemento =  nro_nans_i, nro_max_de_caracteres = max_cifras_nan)
   
    tipo_de_dato   = df_car[i].dtype
    tipo_dato_impr = impresor_de_caracteres(elemento = tipo_de_dato, nro_max_de_caracteres = max_cifras_tipo_de_dato)  
   
    variable_i     = i
    vari_impr      = impresor_de_caracteres(elemento =  variable_i, nro_max_de_caracteres = max_cifras_variable)

    contador_i     = cont + 1
    cont_impr      = impresor_de_caracteres(elemento =  contador_i, nro_max_de_caracteres = max_cifras_contador)

    #print('{}) categorias = {}   |   NaNs = {}   |   Tipo de dato = {}   |   variable = {}'.format(cont_impr, categ_impr,  nans_impr, tipo_dato_impr,      vari_impr))
    print('{}) variable = {}   |   categorias = {}   |   NaNs = {}   |   Tipo de dato = {}'.format(cont_impr,  vari_impr, categ_impr,      nans_impr, tipo_dato_impr))


def cardinalidad_spark(df_car):

  encabezados = []
  nro_filas = df_car.count()
  nro_cols  = len(df_car.columns)

  nro_datos_repetidos = nro_filas - df_car.dropDuplicates().count()

  print('NRO DE COLUMNAS        = {}'.format(nro_cols))
  print('NRO DE FILAS           = {}'.format(nro_filas))
  print('NRO DE FILAS REPETIDAS = {}'.format(nro_datos_repetidos))

  tipos_de_datos = []
  for i in df_car.dtypes:
    tipos_de_datos.append(i[1])
    encabezados.append(i[0])

  max_cifras_cat          = 8
  max_cifras_nan          = 8
  max_cifras_tipo_de_dato = max_cantidad_de_caractares_en_listado(listado = tipos_de_datos)
  max_cifras_variable     = max_cantidad_de_caractares_en_listado(listado = encabezados)
  max_cifras_contador     = len(str(len(encabezados)))

  for cont, i in enumerate(encabezados):
  
    nro_categ_i    = df_car.select(col(i)).distinct().count()
    #nro_categ_i    = len(pd.unique(df_car[i]))
    categ_impr     = impresor_de_caracteres(elemento = nro_categ_i, nro_max_de_caracteres = max_cifras_cat)   
    
    nro_nans_i     = df_car.filter(col(i).isNull()).count()
    nans_impr      = impresor_de_caracteres(elemento =  nro_nans_i, nro_max_de_caracteres = max_cifras_nan)
   
    tipo_de_dato   = tipos_de_datos[cont]
    tipo_dato_impr = impresor_de_caracteres(elemento = tipo_de_dato, nro_max_de_caracteres = max_cifras_tipo_de_dato)  
   
    variable_i     = i
    vari_impr      = impresor_de_caracteres(elemento =  variable_i, nro_max_de_caracteres = max_cifras_variable)

    contador_i     = cont + 1
    cont_impr      = impresor_de_caracteres(elemento =  contador_i, nro_max_de_caracteres = max_cifras_contador)

    #print('{}) categorias = {}   |   NaNs = {}   |   Tipo de dato = {}   |   variable = {}'.format(cont_impr, categ_impr, nans_impr, tipo_dato_impr, vari_impr))
    print('{}) variable = {}   |   categorias = {}   |   NaNs = {}   |   Tipo de dato = {}'.format(cont_impr,  vari_impr, categ_impr,      nans_impr, tipo_dato_impr))

# -------------------------------------------------------------------------
# ---------------- PARA MIRAR LA CARDINALIDAD DE UNA COLUMNA --------------
# -------------------------------------------------------------------------

def value_counts_plus(dataframe, var):
    df_value_counts      = dataframe[var].value_counts()
    clases               = df_value_counts.keys().tolist()
    valores              = df_value_counts.values.tolist()
    valores_porcent      = dataframe[var].value_counts(normalize = True).values
    valores_porcent_acum = acumulador_porcentaje(valores_porcent)

    nro_decimales = 3
    porcentaje    = True
    valores_porcent      = list(map( lambda valor: configurar_nro_decimal(valor, nro_decimales, porcentaje), valores_porcent     ))
    valores_porcent_acum = list(map( lambda valor: configurar_nro_decimal(valor, nro_decimales, porcentaje), valores_porcent_acum))

    max_cifras_valores      = 6
    max_cifras_clases       = max_cantidad_de_caractares_en_listado(listado = clases              )
    max_cifras_porcent      = max_cantidad_de_caractares_en_listado(listado = valores_porcent     )
    max_cifras_porcent_acum = max_cantidad_de_caractares_en_listado(listado = valores_porcent_acum)
    max_cifras_contador     = len(str(len(clases)))

    print('<<<<<<<<< ' + var + ' >>>>>>>>>\n' )
    print('CANTIDAD TOTAL DE CLASES = ' + str(len(clases)) )
    for cont, i in enumerate(clases):
        
        nro_valores       = valores[cont]
        valor_impr        = impresor_de_caracteres(elemento = nro_valores,      nro_max_de_caracteres = max_cifras_valores     )

        nro_porcent       = valores_porcent[cont]
        porcent_impr      = impresor_de_caracteres(elemento = nro_porcent,      nro_max_de_caracteres = max_cifras_porcent     )

        nro_porcent_acum  = valores_porcent_acum[cont]
        porcent_acum_impr = impresor_de_caracteres(elemento = nro_porcent_acum, nro_max_de_caracteres = max_cifras_porcent_acum)

        variable_i        = i
        vari_impr         = impresor_de_caracteres(elemento = variable_i,       nro_max_de_caracteres = max_cifras_clases      )

        contador_i        = cont + 1
        cont_impr         = impresor_de_caracteres(elemento = contador_i,       nro_max_de_caracteres = max_cifras_contador    )
    
        print('{}) cantidad = {}   |   cantidad_porcentual = {} %  |  {} %  |   variable = {}'.format(cont_impr, valor_impr, porcent_impr, porcent_acum_impr, vari_impr))

    print('---------------------------------------------------------------------------------------------------------------------------------------------------------------')


def value_counts_spark_individual(dataframe,var):
  df_conteo    = dataframe.groupby(var).agg(count(var).alias('conteo'))
  total        = df_conteo.select(sum('conteo')).collect()[0][0]
  df_resultado = df_conteo.withColumn('porcentaje', expr('conteo / {}'.format(total)))
  df_resultado = df_resultado.sort(desc('conteo'))
  return df_resultado

def value_counts_plus_spark(dataframe, var):
    df_value_counts      = value_counts_spark_individual(dataframe,var)
    clases               = df_value_counts.select(var).rdd.flatMap(lambda x: x).collect()
    valores              = df_value_counts.select('conteo').rdd.flatMap(lambda x: x).collect()
    valores_porcent      = df_value_counts.select('porcentaje').rdd.flatMap(lambda x: x).collect()
    valores_porcent_acum = acumulador_porcentaje(valores_porcent)

    nro_decimales = 3
    porcentaje    = True
    valores_porcent      = list(map( lambda valor: configurar_nro_decimal(valor, nro_decimales, porcentaje), valores_porcent     ))
    valores_porcent_acum = list(map( lambda valor: configurar_nro_decimal(valor, nro_decimales, porcentaje), valores_porcent_acum))

    max_cifras_valores      = 8
    max_cifras_clases       = max_cantidad_de_caractares_en_listado(listado = clases              )
    max_cifras_porcent      = max_cantidad_de_caractares_en_listado(listado = valores_porcent     )
    max_cifras_porcent_acum = max_cantidad_de_caractares_en_listado(listado = valores_porcent_acum)
    max_cifras_contador     = len(str(len(clases)))

    print('<<<<<<<<< ' + var + ' >>>>>>>>>\n' )
    print('CANTIDAD TOTAL DE CLASES = ' + str(len(clases)) )
    for cont, i in enumerate(clases):
        
        nro_valores       = valores[cont]
        valor_impr        = impresor_de_caracteres(elemento = nro_valores,      nro_max_de_caracteres = max_cifras_valores     )

        nro_porcent       = valores_porcent[cont]
        porcent_impr      = impresor_de_caracteres(elemento = nro_porcent,      nro_max_de_caracteres = max_cifras_porcent     )

        nro_porcent_acum  = valores_porcent_acum[cont]
        porcent_acum_impr = impresor_de_caracteres(elemento = nro_porcent_acum, nro_max_de_caracteres = max_cifras_porcent_acum)

        variable_i        = i
        vari_impr         = impresor_de_caracteres(elemento = variable_i,       nro_max_de_caracteres = max_cifras_clases      )

        contador_i        = cont + 1
        cont_impr         = impresor_de_caracteres(elemento = contador_i,       nro_max_de_caracteres = max_cifras_contador    )
    
        print('{}) cantidad = {}   |   cantidad_porcentual = {} %  |  {} %  |   variable = {}'.format(cont_impr, valor_impr, porcent_impr, porcent_acum_impr, vari_impr))

    print('---------------------------------------------------------------------------------------------------------------------------------------------------------------')

# -------------------------------------------------------------------------
# PARA MIRAR CARDINALIDAD DE UNA COLUMNA DE LOS VALORES NAN DE OTRA COLUMNA
# -------------------------------------------------------------------------

def value_counts_nans(dataframe, col_con_nan, col_clases_a_mirar):
  #clases_con_nan       = dataframe[dataframe[col_con_nan].isna()][col_clases_a_mirar].unique().tolist()
  serie_clases_con_nan = dataframe.groupby(col_clases_a_mirar)[col_con_nan].apply(lambda datos_agrupados: datos_agrupados.isna().sum())
  #print(serie_clases_con_nan[clases_con_nan])
  print(serie_clases_con_nan)


def value_counts_nans_spark(dataframe, col_con_nan, col_clases_a_mirar):
  #clases_con_nan       = dataframe.filter( col(col_con_nan).isNull() ).select(col_clases_a_mirar).distinct().rdd.flatMap(lambda x: x).collect()
  df_nans_en_columna = dataframe.groupby(col_clases_a_mirar).agg( ( sum(col(col_con_nan).isNull().cast("int")) ) .alias("num_nans") )
  return df_nans_en_columna

# -----------------------------------------------------------------------------
# -------------------- PARA CREAR UN MAPA DE CORRELACIONES -------------------- 
# -----------------------------------------------------------------------------

def mapa_correlaciones(lista_df, lista_vars, nro_columnas_subplot, figsize_subplots, tamanio_fuente = 0.8, dims_hor =10,dims_vert=5, mostrar_letreros_hor=True):
  lista_df_matriz_correlaciones = []
  for df_i in lista_df:
    lista_df_matriz_correlaciones.append(df_i.corr())
  imprimir_matriz_correlaciones(lista_df_matriz_correlaciones, lista_vars, nro_columnas_subplot,  figsize_subplots,  tamanio_fuente, dims_hor ,dims_vert, mostrar_letreros_hor)


def mapa_correlaciones_spark(dataframe, dims_vert = 5, dims_hor = 5, tamanio_fuente = 0.8):
  df_matriz_correlaciones = matriz_correlaciones_spark(dataframe)
  imprimir_matriz_correlaciones(df_matriz_correlaciones, dims_vert = dims_vert, dims_hor = dims_hor, tamanio_fuente = tamanio_fuente)



def imprimir_matriz_correlaciones(lista_corr_matriz, lista_vars, nro_columnas_subplot, figsize_subplots, tamanio_fuente, dims_hor ,dims_vert, mostrar_letreros_hor):
    encabezados_nuevos = lista_vars
    nro_de_variables       = len(lista_vars)
    contador_graficas  = 0
    tam                = nro_de_variables#np.sum(range(1,nro_de_variables)) if nro_de_variables > 1 else nro_de_variables - 1    # Esta suma es para determinar el número de gráficos que se deben considerar en TOTAL
    filas              = int(tam / nro_columnas_subplot) if (tam % nro_columnas_subplot) == 0 else int(tam / nro_columnas_subplot) + 1                                   ########################################

    figure, axis = plt.subplots(nrows = filas, ncols = nro_columnas_subplot,figsize=figsize_subplots )
    axis = axis.flatten() if nro_columnas_subplot >1 else [axis]

    for cont_i,encabezado_i in enumerate(encabezados_nuevos):                                                                                                                #########################################
                sns.set_theme(font_scale=tamanio_fuente)
                plt.figure(figsize=(dims_hor,dims_vert))
                heat_map = sns.heatmap(lista_corr_matriz[contador_graficas], linewidths=.05, cmap = 'RdBu', vmin = -1, vmax =1, annot = True, ax = axis[contador_graficas], xticklabels=mostrar_letreros_hor)
                axis[contador_graficas].set_title(encabezado_i, color='gray')
                contador_graficas += 1
    plt.tight_layout();
   

def matriz_correlaciones_spark(dataframe):

  cols_NO_numericas  = ['string', 'date', 'boolean']
  tipos_de_datos     = dataframe.dtypes
  columnas_numericas = [i[0] for i in tipos_de_datos if i[1] not in cols_NO_numericas ]
  #print(columnas_numericas)
  dataframe          = dataframe.select(*columnas_numericas)

  columnas = dataframe.columns
  dimensiones = len(columnas)
  matriz = np.ones([dimensiones,dimensiones])

  for cont_i, col_i in enumerate(columnas):
     for col_j in columnas[cont_i:]:
        cont_j = columnas.index(col_j)
        correlacion_i_j = dataframe.select(corr(col_i, col_j)).collect()[0][0]
        matriz[cont_i][cont_j] = correlacion_i_j
        matriz[cont_j][cont_i] = correlacion_i_j
  
  df_matriz_correlaciones_pandas = pd.DataFrame(matriz, columns = columnas_numericas, index = columnas_numericas)
  return df_matriz_correlaciones_pandas

# -----------------------------------------------------------------------------
# ---- PARA CREAR UN DF CON LAS PRINCIPALES CORRELACIONES DE CADA VARIABLE ----
# -----------------------------------------------------------------------------

def rankeador_corr(dataframe):
    df_matriz_correlaciones = dataframe.corr()
    vars_dataframe          = df_matriz_correlaciones.columns.tolist()
    df_ranking              = organizador_ranking(df_matriz_correlaciones,vars_dataframe)
    return df_ranking


def rankeador_corr_spark(dataframe):
    df_matriz_correlaciones = matriz_correlaciones_spark(dataframe)
    vars_dataframe          = df_matriz_correlaciones.columns
    df_ranking              = organizador_ranking(df_matriz_correlaciones,vars_dataframe)
    return df_ranking


def organizador_ranking(df_matriz_correlaciones,vars_dataframe):
    df_ranking  = pd.DataFrame()
    for var_i in vars_dataframe:
        df_aux            = df_matriz_correlaciones[var_i].sort_values(ascending = False).copy()
        indices           = df_aux.index
        valores           = df_aux.values
        datos_rankeados   = list(map( lambda x,y : x + ' | (' + str(round(y,3)) + ')', indices, valores))
        df_ranking[var_i] = datos_rankeados
    return df_ranking

# ------------------------------------------------------------------------------
# -- CREAR UN DF CON LAS PRINCIPALES CORRELACIONES DE CADA VARIABLE COLOREADO --
# ------------------------------------------------------------------------------

def dataframe_correlacion_grafico (dataframe, variables_a_comparar, variables_a_dejar_en_filas = False, variable_para_ordenar = False, umbral_positivo = 0, umbral_negativo = 0):
  df_matriz_correlaciones = dataframe.corr()
  df_matriz_correlaciones = df_matriz_correlaciones[variables_a_comparar]
  salida                  = graficar_dataframe_correlaciones(df_matriz_correlaciones, variables_a_comparar, variables_a_dejar_en_filas, variable_para_ordenar, umbral_positivo, umbral_negativo)
  return salida


def dataframe_correlacion_grafico_spark (dataframe, variables_a_comparar, variables_a_dejar_en_filas = False, variable_para_ordenar = False, umbral_positivo = 0, umbral_negativo = 0):
  df_matriz_correlaciones = matriz_correlaciones_spark(dataframe)
  salida                  = graficar_dataframe_correlaciones(df_matriz_correlaciones, variables_a_comparar, variables_a_dejar_en_filas, variable_para_ordenar, umbral_positivo, umbral_negativo)
  return salida


def graficar_dataframe_correlaciones(df_matriz_correlaciones, variables_a_comparar, variables_a_dejar_en_filas = False, variable_para_ordenar = False, umbral_positivo = 0, umbral_negativo = 0):
  df_matriz_correlaciones = df_matriz_correlaciones[variables_a_comparar]
  #cm                      = sns.light_palette('green', as_cmap = True)
  cm                      = sns.light_palette('RdBu', as_cmap = True)
  if variables_a_dejar_en_filas  != False:
    df_matriz_correlaciones = df_matriz_correlaciones.filter(items = variables_a_dejar_en_filas, axis = 0)

  if variable_para_ordenar != False:
    df_matriz_correlaciones.sort_values(variable_para_ordenar, ascending = False, inplace = True)
    if (umbral_positivo > 0) & (umbral_negativo < 0):
      limite_de_correlaciones(df_matriz_correlaciones, variable_para_ordenar, umbral_positivo , umbral_negativo)
      df_matriz_correlaciones = df_matriz_correlaciones[   ((df_matriz_correlaciones[variable_para_ordenar] > umbral_positivo) & (df_matriz_correlaciones[variable_para_ordenar] < 1))   |   ((df_matriz_correlaciones[variable_para_ordenar] < umbral_negativo) & (df_matriz_correlaciones[variable_para_ordenar] > -1))   ]
    
  salida = df_matriz_correlaciones.style.background_gradient(cmap = cm)
  return salida

# ------------------------------------------------------------------------------
# -- PARA GRAFICAR LAS CORRELACIONES  DE UNA SOLA VARIABLE Y SABER CUAL SIRVE --
# ------------------------------------------------------------------------------

def limite_de_correlaciones(df_corr, variable_para_ordenar, lim_positivo, lim_negativo):
  dims_x = 40
  dims_y = 4
  plt.subplots(figsize = (dims_x, dims_y))
  plt.plot( df_corr [ df_corr[variable_para_ordenar] < 1] )
  plt.plot( [ lim_positivo]*len(df_corr.index), c = 'red', linestyle = (0, (1, 2, 5, 5)), linewidth = 2 )
  plt.plot( [          0.0]*len(df_corr.index), c =   'k', linestyle = (0, (1, 1))      , linewidth = 1 )
  plt.plot( [ lim_negativo]*len(df_corr.index), c = 'red', linestyle = (0, (1, 2, 5, 5)), linewidth = 2 )
  plt.tick_params(axis = 'x', labelrotation = 90, labelsize = 10)
  plt.grid()

# ------------------------------------------------------------------------------
# ----------------------------- CLASES PARETO ----------------------------------
# ------------------------------------------------------------------------------

def clases_pareto(dataframe, var, palabra_para_remplazo = 'otros', porcentaje_pareto = 0.8):
    '''
    Función para reduir variables categóricas con cardinalidad alta, tomando/seleccionan-
    do las que aporten/aparezcan un mayor porcentaje (por defecto 80%) y modificando el da-
    taframe para que solo se dejen dichas categorías y en las que menos porcentaje aporten,
    aparezca la palabra 'otros'

    *inputs:
        - dataframe
        - var: String con el nombre de la columna a trabajar
        - porcentaje_pareto: Número decimal de peso acumulado (0.8 por defecto)
    '''

    df_value_counts         = pd.DataFrame(dataframe[var].value_counts())
    valores_porcent         = dataframe[var].value_counts(normalize = True).values
    valores_porcent_acum    = acumulador_porcentaje(valores_porcent)
    df_value_counts['acum'] = valores_porcent_acum
    df_aux                  = df_value_counts[df_value_counts['acum'] >= porcentaje_pareto].copy()

    clases_importantes        = df_aux.index.tolist()
    clases_unicas_originales  = dataframe[var].unique()

    clases_unicas_remplazadas = [ i if i in clases_importantes else palabra_para_remplazo for i in clases_unicas_originales ]

    #print(clases_importantes)
    #print(clases_unicas_originales)
    #print(clases_unicas_remplazadas)

    encoder_custom(var = var, dataframe = dataframe, arr_anterior = clases_unicas_originales, \
                         arr_nuevo = clases_unicas_remplazadas, arr_nuevo_num = False)


def clases_pareto_spark(dataframe, var, palabra_para_remplazo = 'otros', porcentaje_pareto = 0.8):
    '''
    Función para reduir variables categóricas con cardinalidad alta, tomando/seleccionan-
    do las que aporten/aparezcan un mayor porcentaje (por defecto 80%) y modificando el da-
    taframe para que solo se dejen dichas categorías y en las que menos porcentaje aporten,
    aparezca la palabra 'otros'

    *inputs:
        - dataframe
        - var: String con el nombre de la columna a trabajar
        - porcentaje_pareto: Número decimal de peso acumulado (0.8 por defecto)
    '''

    df_value_counts_spark   = value_counts_spark_individual(dataframe,var)
    df_value_counts         = df_value_counts_spark.toPandas()
    df_value_counts.index   = df_value_counts[var]
    del df_value_counts[var]
    #print(df_value_counts)

    valores_porcent         = df_value_counts['porcentaje']#dataframe[var].value_counts(normalize = True).values
    valores_porcent_acum    = acumulador_porcentaje(valores_porcent)
    df_value_counts['acum'] = valores_porcent_acum
    df_aux                  = df_value_counts[df_value_counts['acum'] >= porcentaje_pareto].copy()

    clases_importantes        = df_aux.index.tolist()
    clases_unicas_originales  = dataframe.select(col(var)).distinct().rdd.flatMap(lambda x : x).collect()

    clases_unicas_remplazadas = [ i if i in clases_importantes else palabra_para_remplazo for i in clases_unicas_originales ]

    df_transformado = encoder_custom_spark(var = var, dataframe = dataframe, arr_anterior = clases_unicas_originales, \
                         arr_nuevo = clases_unicas_remplazadas, arr_nuevo_num = False)
    return df_transformado


def seleccionador_de_valor(dato, limites):
    #print(dato)
    for cont in range(1,len(limites)):
            #limi_inf = limites[cont-1] - abs(limites[cont-1]*0.02) if cont == 1            else limites[cont-1]
            #limi_sup = limites[cont  ] + abs(limites[cont  ]*0.02) if cont == len(limites) else limites[cont  ]
            condicion = (limites[cont-1] <= dato) if cont == 1 else (limites[cont-1] < dato)
            #if ( limi_inf < dato) & ( dato <= limi_sup):
            if  condicion & ( dato <= limites[cont]):
                dato_nuevo = str(limites[cont-1]) + ' - ' + str(limites[cont])
                #print(str(dato)+ ' - ' +dato_nuevo )                
    return dato_nuevo



def discretizador(dataframe, var):
  params   = plt.hist(dataframe[var])
  altura   = params[0]
  lims_x   = params[1]
  dataframe[var + '_discreto'] = dataframe[var].apply(lambda x: seleccionador_de_valor(dato = x, limites = lims_x))

def discretizdor_spark(dataframe, var, bins = 10, arr_nuevo_num = True):
    inicio  = dataframe.agg({var:'min'}).collect()[0][0]
    fin     = dataframe.agg({var:'max'}).collect()[0][0]
    bins    = 10
    paso    = ( fin - inicio ) / bins
    limites = [ paso*i+inicio for i in range(bins) ] + [fin]

    var_nueva = var + '_discreta'
    dataframe = dataframe.withColumn(var_nueva, col(var))


    for i in range(len(limites)-1):  
        dato =  (limites[ i]+limites[ i+1]) / 2 if arr_nuevo_num else str(limites[ i])+'-'+ str(limites[ i+1])
        dataframe = dataframe.withColumn( var_nueva, when( (limites[ i] <= col(var)) & (col(var) <= limites[ i+1]) , dato ).otherwise(col(var_nueva)) )

    if arr_nuevo_num:
        dataframe = dataframe.withColumn( var_nueva, col(var_nueva).cast('float'))
    else:
        dataframe = dataframe.withColumn( var_nueva, col(var_nueva).cast('string'))

    return dataframe

# -------------------------------------------------------------------------
# ------------------------ ENCODER PERSONALIZADO --------------------------
# -------------------------------------------------------------------------

def encoder_custom(var, dataframe, arr_anterior, arr_nuevo, arr_nuevo_num = True):
    '''
    Función para hacer el cambio de las categorías de una variable, por otras categorías
    (pueden ser números), definidas previamente por el usuario. Esta función modificará di-
    rectamente el dataframe

    *inputs:
        - dataframe
        - var: String con el nombre de la columna a trabajar
        - arr_anterior: valores unicos de lascategorias originales (de la variable "var")
        - arr_nuevo: valores únicos de las categorías que el usuario desea que remplacen a
          las categorias contenidas en "arr_anterior". Ambas "arr_anterior" y  "arr_nuevo"
          deben estar en el mismo orden y por ende tener la misma longitud
        - arr_nuevo_num: Será "True" si "arr_nuevo" está totalmente conformado por valores
          numericos, de lo contrario debe ser puesto en "False"
    '''
    var_nueva = var + '_custom_enc'
    dataframe[var_nueva] = dataframe[var]
    for i in range(len(arr_anterior)):
        dataframe[var_nueva] = np.where(dataframe[var_nueva] == arr_anterior[i], str(arr_nuevo[i]), dataframe.astype({var_nueva: 'object'})[var_nueva])
    dataframe[var_nueva] = dataframe.astype({var_nueva: 'int32'})[var_nueva] if arr_nuevo_num else dataframe.astype({var_nueva: 'object'})[var_nueva]

def encoder_custom_spark(var, dataframe, arr_anterior, arr_nuevo, arr_nuevo_num = True):
    '''
    Función para hacer el cambio de las categorías de una variable, por otras categorías
    (pueden ser números), definidas previamente por el usuario. Esta función NO modificará di-
    rectamente el dataframe

    *inputs:
        - dataframe
        - var: String con el nombre de la columna a trabajar
        - arr_anterior: valores unicos de lascategorias originales (de la variable "var")
        - arr_nuevo: valores únicos de las categorías que el usuario desea que remplacen a
          las categorias contenidas en "arr_anterior". Ambas "arr_anterior" y  "arr_nuevo"
          deben estar en el mismo orden y por ende tener la misma longitud
        - arr_nuevo_num: Será "True" si "arr_nuevo" está totalmente conformado por valores
          numericos, de lo contrario debe ser puesto en "False"
    '''
    var_nueva = var + '_custom_enc'
    dataframe = dataframe.withColumn(var_nueva, col(var))

    for i in range(len(arr_anterior)):
        dataframe = dataframe.withColumn( var_nueva, when(col(var_nueva) == arr_anterior[i], str(arr_nuevo[i])).otherwise(col(var_nueva)) )

    if arr_nuevo_num:
        dataframe = dataframe.withColumn( var_nueva, col(var_nueva).cast('int'))
    else:
        dataframe = dataframe.withColumn( var_nueva, col(var_nueva).cast('string'))

    #dataframe.show()
    return dataframe

def encoder_fecha_a_num(dataframe, variables_fechas, retornar_nombres_cols_transformadas = False):
    
    for cont, var_fecha_i in enumerate(variables_fechas):
        if cont == 0:
            fecha_min = np.min(dataframe[var_fecha_i])
        else:
            fecha_min_i = np.min(dataframe[var_fecha_i])
            if fecha_min_i < fecha_min:
                fecha_min = fecha_min_i
    
    nombres_cols_transf = []
    for col_i in variables_fechas:
        nombre = col_i+'_fecha_enc'
        nombres_cols_transf.append(nombre)
        dataframe[nombre] = dataframe[col_i].apply(lambda x: (x - fecha_min).days)
    
    if retornar_nombres_cols_transformadas == True:
      return  nombres_cols_transf


def label_encoder_plus(dataframe, variables):
    arr_objts = []
    for var_i in variables:
        #print(var_i)
        objeto_labelEnc_i             = LabelEncoder()
        dataframe[var_i+'_label_enc'] = objeto_labelEnc_i.fit_transform(dataframe[var_i])
        arr_objts.append(objeto_labelEnc_i)
    return arr_objts


def encoder_dummies_plus(dataframe, variables):
    df_dum    = pd.get_dummies(dataframe, columns=variables)#, prefix=["Type_is"] )
    cols_elim = [i for i in dataframe.columns.tolist() if i not in variables]
    for i in cols_elim:
        del df_dum[i]
    return dataframe.join(df_dum)

# ---------------------------------------------------------------------------------------------------------------
# ----------REMPLAZAR TODO UN DATO SI SE ENCUENTRA UN PATRON DE CARACTERES USANDO EXPRESIONES REGULARES----------
# ---------------------------------------------------------------------------------------------------------------

#def ajuste_er_remplazo_total(dato, conjunto_er):
#  for valor_a_dejar, valor_posible_er in conjunto_er:
#    dato = valor_a_dejar if re.search(valor_posible_er , dato) else dato
#  return dato

# ----------------------------------------------------------------------------------------------------------------
# ---REMPLAZAR LA COINCIDENCIA DE UN DATO SI SE ENCUENTRA UN PATRON DE CARACTERES USANDO EXPRESIONES REGULARES---
# ----------------------------------------------------------------------------------------------------------------

def ajuste_er(dato, diccionario_er):
  for valor_a_dejar, valor_er in diccionario_er.items():
    for er_i in valor_er:
      dato = re.sub(er_i, valor_a_dejar, dato)
  return dato

def ajuste_er_remplazo_caracteres(dataframe, columna, diccionario_er):
   dataframe[columna] = dataframe[columna].apply(lambda dato : ajuste_er(dato,diccionario_er))
   return dataframe

def ajuste_er_remplazo_caracteres_spark(dataframe, columna, diccionario_er):
  for valor_a_dejar, valor_posible_er in diccionario_er.items():
      dataframe = dataframe.withColumn(columna, regexp_replace(col(columna), valor_posible_er, valor_a_dejar))
  return dataframe

# ----------------------------------------------------------------------------------------------------------------
# ------------------ FUNCIÓN PARA CASTEAR LOS TIPOS DE DATOS DE LAS COLUMNAS DE UN DATAFRAME ---------------------
# ----------------------------------------------------------------------------------------------------------------

def caster(dataframe, list_cols_cat = False, list_cols_int = False, list_cols_float = False, list_cols_fecha = False, list_cols_bool = False):
    if list_cols_cat != False:
        for col_cat_i in list_cols_cat:
            dataframe[col_cat_i] = dataframe[col_cat_i].astype(str)
   
    if list_cols_int != False:
        for col_int_i in list_cols_int:
            dataframe[col_int_i] = dataframe[col_int_i].astype(int)

    if list_cols_float != False:
        for col_float_i in list_cols_float:
            dataframe[col_float_i] = dataframe[col_float_i].astype(float)

    if list_cols_fecha != False:
        for col_fecha_i in list_cols_fecha:
            dataframe[col_fecha_i] = pd.to_datetime(dataframe['fecha'])

    if list_cols_bool != False:
        for col_bool_i in list_cols_bool:
            dataframe[col_bool_i] = dataframe[col_bool_i].astype(bool)
    return dataframe

def caster_spark(dataframe, list_cols_cat = False, list_cols_int = False, list_cols_float = False, list_cols_fecha = False, list_cols_bool = False):
    if list_cols_cat != False:
        for col_cat_i in list_cols_cat:
            dataframe = dataframe.withColumn(col_cat_i, expr("CAST(`"+col_cat_i+"` AS STRING)"))
   
    if list_cols_int != False:
        for col_int_i in list_cols_int:
            dataframe = dataframe.withColumn(col_int_i,   col(col_int_i).cast("int"))

    if list_cols_float != False:
        for col_float_i in list_cols_float:
            dataframe = dataframe.withColumn(col_float_i, col(col_float_i).cast("float"))

    if list_cols_fecha != False:
        for col_fecha_i in list_cols_fecha:
            dataframe = dataframe.withColumn(col_fecha_i, to_date(col(col_fecha_i), 'yyyy-MM-dd'))

    if list_cols_bool != False:
        for col_bool_i in list_cols_bool:
            dataframe = dataframe(col_bool_i, when(col(col_bool_i) == 'true', True).otherwise(False))

    return dataframe

######################################################################################################################
############################################ FUNCIONES PARA GRAFICAR #################################################
######################################################################################################################


# -------------------------------------------------------------------------
# ------------------------ DIAGRAMAS DE DISPERSIÓN ------------------------
# -------------------------------------------------------------------------


def especificaciones_dispersor(ax, var_x, var_y):
    #axis[contador_graficos].tick_params(labelsize = 7)
    #axis[contador_graficos].set_title(i+'     VS     '+j, fontsize=15)    # Fijamos un título par cada gráfica
    ax.set_xlabel(var_x, fontsize=15)                                      # Fijamos el titulo del eje X
    ax.set_xticks([])                                                      # Apagamos las etiquetas del eje X
    ax.set_ylabel(var_y, fontsize=15)                                      # Fijamos el titulo del eje Y
    ax.set_yticks([])                                                      # Apagamos las etiquetas del eje Y
    #axis[contador_graficos].axes.xaxis.set_visible(False)                 # Con esto ocultamos TODA la información del eje X
    #axis[contador_graficos].axes.yaxis.set_visible(False)                 # Con esto ocultamos TODA la información del eje Y


def imprimir_colores_clases(clases_a_imprimir,colores_a_imprimir):
    for cl, co in zip(clases_a_imprimir, colores_a_imprimir):
        print(co,' - ' ,cl)
    

def dispersor_clase(dataframe, variable_clase, nro_columnas_subplot, cols_no_graficables, figsize_subplots, colores_categorias, variable_analisis_unico = False):

    encabezados_nuevos = dataframe.columns.tolist()
    encabezados_nuevos = [i for i in encabezados_nuevos if i not in cols_no_graficables]

    solo_variable_clase    = False if variable_analisis_unico == False else True                               ######################################
    encabezados_a_graficar = encabezados_nuevos if solo_variable_clase == False else [variable_analisis_unico] ######################################
    nro_de_variables       = len(encabezados_nuevos)
    
    if variable_clase  != False:
        clases         = dataframe[variable_clase].unique().tolist()  # Si no se definieron clases para pintar, no se usa la variable "clases"
        imprimir_colores_clases(clases, colores_categorias)
    
    contador_graficas  = 0
    tam                = sum(range(1,nro_de_variables)) if solo_variable_clase == False else nro_de_variables - 1    # Esta suma es para determinar el número de gráficos que se deben considerar en TOTAL
    filas              = int(tam / nro_columnas_subplot) if (tam % nro_columnas_subplot) == 0 else int(tam / nro_columnas_subplot) + 1                                   ########################################

    figure, axis = plt.subplots(nrows =filas, ncols = nro_columnas_subplot, figsize = figsize_subplots)
    axis = axis.flatten()

    cant_colores = len(colores_categorias)
    #for cont_i,i in enumerate(encabezados_nuevos):
    for cont_i,i in enumerate(encabezados_a_graficar):                                                                                                                   #########################################
        #for j in encabezados_nuevos[(cont_i+1):]:
        graficas_a_realizar = encabezados_nuevos[(cont_i+1):] if solo_variable_clase == False else [k for k in encabezados_nuevos if k not in [variable_analisis_unico]] #########################################
        for j in graficas_a_realizar:                                                                                                                                    #########################################
            if i != j:
                if cant_colores >0:     
                    for k, color_i in zip(clases,colores_categorias):
                        df_aux = dataframe[dataframe[variable_clase] == k]
                        axis[contador_graficas].scatter(df_aux[i], df_aux[j], color = color_i, s = 200, linewidths=1, marker='o',edgecolors='w', alpha = 0.2)  # Con "o" y con "w" hacemos que los datos se grafiquen como puntos de borde blanco (white)
                        especificaciones_dispersor(ax = axis[contador_graficas],var_x = i,var_y = j)
                        
                else:
                    axis[contador_graficas].scatter(dataframe[i], dataframe[j], s = 200, linewidths=1, marker='o', alpha = 0.2, c = 'k') # ,edgecolors='w'
                    especificaciones_dispersor(ax = axis[contador_graficas], var_x = i,var_y = j)
                contador_graficas += 1
                #print(suma)
    plt.tight_layout();


# -------------------------------------------------------------------------
# ------------------ HISTOGRAMAS DE VARIAS VARIABLES ----------------------
# -------------------------------------------------------------------------


def histogrameador(dataframe, nro_columnas_subplot, cols_no_graficables, figsize_subplots, variable_clases = [], porcentajes = False):
    encabezados_nuevos  = dataframe.columns.tolist()
    encabezados_nuevos  = [i for i in encabezados_nuevos if i not in cols_no_graficables + variable_clases]  
    col_para_histograma = encabezados_nuevos
    tam                 = len(col_para_histograma)
    filas               = int(tam / nro_columnas_subplot) if (tam % nro_columnas_subplot) == 0 else int(tam / nro_columnas_subplot) + 1

    fig, ax = plt.subplots(ncols = nro_columnas_subplot, nrows = filas, figsize = figsize_subplots)
    ax      = ax.flatten()
    cont    = 0
    df_aux  = pd.DataFrame()
    for variable_i in col_para_histograma:
        #print(i)
        if len(variable_clases) == 0:
          df_aux = dataframe.copy()
          graficador(axis = ax[cont], df_a_graficar = df_aux, variable_a_graficar = variable_i, porcentajes = porcentajes)
        else:
          clases = dataframe[variable_clases[0]].unique()
          #print(clases)
          for j in clases:
            df_aux = dataframe[dataframe[variable_clases[0]] == j]
            graficador(axis = ax[cont], df_a_graficar = df_aux, variable_a_graficar = str(variable_i), porcentajes = porcentajes, clase_a_graficar = j) # str(variable_i)+' - '+str(j)


        cont += 1
    plt.tight_layout();
  
def graficador(axis, df_a_graficar, variable_a_graficar, porcentajes, clase_a_graficar = ''):
        axis.hist(df_a_graficar[variable_a_graficar], label = str(variable_a_graficar) + ' - ' + str(clase_a_graficar), density = porcentajes, alpha = 0.2) #bins = 10,
        axis.legend(loc="best", fontsize=20)
        #axis.set_yscale('log')
        axis.tick_params(axis='x', labelrotation=90, labelsize=15)
        axis.tick_params(axis='y', labelrotation=90, labelsize=15)
        #axis.set_xlabel(i)
        #axis.set_title(i)


# -------------------------------------------------------------------------
# ---------------------- BARRAS DE VARIAS VARIABLES -----------------------
# -------------------------------------------------------------------------


# _________________________________________________________________________
# --(vars categóricas de MI interes VS vars numéricas agrupadas por suma)--

def barreador(dataframe, nro_columnas_subplot, cols_no_graficables, figsize_subplots, variables, acciones,
              impr_valores = True, angulo_rotacion_letrero = 0, notacion_cientifica = True, tamanio_valores = 15, tamanio_fuente = 5, nro_decimales=2, imprimir_porcentaje= True):
    encabezados         = dataframe.columns.tolist()
    encabezados_nuevos  = [i for i in encabezados if i not in cols_no_graficables]
    encabezados_nuevos  = [i for i in encabezados_nuevos if i not in variables]
    #tam                 = len(encabezados_nuevos)*len(variables)
    tam                 = len(encabezados_nuevos)*len(variables)*len(acciones)
    filas               = int(tam / nro_columnas_subplot) if (tam % nro_columnas_subplot) == 0 else int(tam / nro_columnas_subplot) + 1

    fig, ax = plt.subplots(ncols = nro_columnas_subplot, nrows = filas, figsize = figsize_subplots) if tam > 1 else plt.subplots(figsize = figsize_subplots)
    ax      = ax.flatten() if tam > 1 else ax
  
    numero_de_datos = len(dataframe)
    cont = 0
    for k in variables:
      var_analisis    = dataframe[k]
      clases          = var_analisis.unique()
      
      for j in encabezados_nuevos:

        for accion in acciones:

          df_aux = pd.DataFrame()
          cantidades = []
          for i in clases:
            df_aux = dataframe[dataframe[k] == i].copy()
            #print(j)
            if   accion == 'suma':
              valor  = sum(df_aux[j])
            elif accion == 'max':
              valor  = max(df_aux[j]) 
            elif accion == 'min':
              valor  = min(df_aux[j])
            elif accion == 'media':
              valor  = df_aux[j].mean()
            elif accion == 'desv_std':
              valor  = df_aux[j].std()
            elif accion == 'mediana':
              valor  = df_aux[j].median()
            elif accion == 'moda':
              valor  = df_aux[j].mode()
            elif accion == 'contar':
              valor  = df_aux[j].count()

            cantidades.append(valor)          
            del df_aux

          try:
            ax_i = ax[cont]
          except:
            ax_i = ax

          if j != k:
            barras = ax_i.bar( clases, cantidades, label = k +' VS '+accion+'('+ j+')', alpha = 0.4) #bins = 10,
            
            ax_i.legend(loc="best", fontsize=tamanio_fuente)
            #ax_i.set_yscale('log')
            ax_i.tick_params(axis='x', labelrotation=90, labelsize=tamanio_fuente)
            ax_i.tick_params(axis='y', labelrotation=90, labelsize=tamanio_fuente)
            #ax_i.set_xlabel(i)
            #ax_i.set_title(k +' VS '+ j, fontsize = 20)
            ax_i.set_yticks([])

            if impr_valores:
              autoetiquetado(barras = barras, axes = ax_i, angulo_rotacion = angulo_rotacion_letrero, 
                             notacion_cientif = notacion_cientifica, tamanio_letreros = tamanio_valores, nro_decimales = nro_decimales, imprimir_porcentaje = imprimir_porcentaje) #, total = numero_de_datos)

            cont += 1
    plt.tight_layout();

def autoetiquetado(barras, axes, angulo_rotacion, notacion_cientif, tamanio_letreros, nro_decimales, imprimir_porcentaje):#, total = 0):
  total   = 0
  alt_max = 0
  for cont, barra_i in enumerate(axes.patches):                   # En este "for" calculamos la altura de la barra mas alta
    altura  = barra_i.get_height()
    total   = altura + total
    alt_max = altura if abs(altura) > abs(alt_max) else alt_max

  #print(alt_max)
  if alt_max < 2:                   # Esta es una medida arbitratia para que los letreros de las gráficas queden relativamente centrados cuando la altura de las barras es muy pequeña
    coordenada_y = alt_max*0.8
  else:
    coordenada_y = alt_max // 2

  for barra_i in axes.patches:#barras:
    altura_barra_i = barra_i.get_height()
    coordenada_x   = barra_i.get_x() + barra_i.get_width() / 2

    if notacion_cientif:
       altura_barra = altura_barra_i
       etiqueta = '{:.2E}'    # cuando esto se toma, se imprimen los resultados en notación científica
    else:
       tupla = ajuste_etiqueta_numero(altura_barra_i, nro_decimales)
       altura_barra = float(tupla[0])
       etiqueta     = tupla[1]

    #print(type(altura_barra), altura_barra_i,end = '-')
    #print(type(etiqueta), etiqueta)
    texto = '\n '+etiqueta.format(altura_barra)+'  \n {:2.2%}'.format(altura_barra_i/total ) if imprimir_porcentaje else '\n '+etiqueta.format(altura_barra)
    texto = texto if total != 0 else '{:.2f}'.format(altura_barra_i)

    axes.annotate(
                  texto,
                  xy         = (coordenada_x, coordenada_y),
                  xytext     = (coordenada_x, coordenada_y),
                  rotation   = angulo_rotacion,
                  #xytext     = (0, -altura_barra_i // 2),  # 3 points vertical offset
                  #textcoords = "offset points",
                  ha         = 'center',
                  va         = 'center',#'bottom',
                  size       = tamanio_letreros,#15,
                  color      = 'k' #'white
                  )

def ajuste_etiqueta_numero(texto, nro_decimales):
    valor = float(texto)
    if abs(valor) < 1e3:
      etiqueta = '{:.'+str(nro_decimales)+'f}'
    elif ( 1e3 <= abs(valor) ) & ( abs(valor) < 1e6 ):
      valor = valor / 1e3
      etiqueta = '{:.'+str(nro_decimales)+'f} Mil'
    elif ( 1e6 <= abs(valor) ) & ( abs(valor) < 1e9 ):
      valor = valor / 1e6
      etiqueta = '{:.'+str(nro_decimales)+'f} Millones'        
    elif ( 1e9 <= abs(valor) ) & ( abs(valor) < 1e12 ):
      valor = valor / 1e9
      etiqueta = '{:.'+str(nro_decimales)+'f} MilMillones'
    elif ( 1e12 <= abs(valor) ) :
      valor = valor / 1e12
      etiqueta = '{:.'+str(nro_decimales)+'f} Billones'
    
    #print(valor, etiqueta)
    #print(type(valor), type(etiqueta))
    return str(valor), etiqueta
# _________________________________________________________________________
# ----------(vars categóricas VS vars categóricas de MI interes)-----------

def barriador_categoricas(dataframe, nro_columnas_subplot, cols_no_graficables, figsize_subplots, variables, una_barra = False, porcentaje = False):
  encabezados         = dataframe.columns.tolist()
  encabezados_nuevos  = [i for i in encabezados if i not in cols_no_graficables]
  encabezados_nuevos  = [i for i in encabezados_nuevos if i not in variables]
  tam                 = len(encabezados_nuevos)
  filas               = int(tam / nro_columnas_subplot) if (tam % nro_columnas_subplot) == 0 else int(tam / nro_columnas_subplot) + 1

  fig, ax = plt.subplots(ncols = nro_columnas_subplot, nrows = filas, figsize = figsize_subplots) if tam > 1 else plt.subplots(figsize = figsize_subplots)
  ax      = ax.flatten() if tam > 1 else ax

  if porcentaje == True:
    normalizador = 'index'
  else:
    normalizador = porcentaje

  for j in variables:
    for i, col in enumerate(encabezados_nuevos):
      try: 
        ax_i = ax[i]
      except:
        ax_i = ax

      pd.crosstab(dataframe[col],dataframe[j], normalize = normalizador).plot(kind='bar', ax=ax_i, stacked = una_barra)
      ax_i.set_xlabel(ajusta_titulo(col), fontsize = 20)
      ax_i.legend(loc="best", fontsize=20)
      #ax_i.set_yscale('log')
      ax_i.tick_params(axis='x', labelrotation=90, labelsize=20)
  plt.tight_layout();


def ajusta_titulo(string):
  if len(string) > 22:
    string = string[:18] + '\n' + string[18:]
  return string

# -------------------------------------------------------------------------
# ------------------------------- BOXPLOTS --------------------------------
# -------------------------------------------------------------------------


def boxploteador(dataframe, nro_columnas_subplot, cols_no_graficables, figsize_subplots):
    encabezados_nuevos  = dataframe.columns.tolist()
    encabezados_nuevos  = [i for i in encabezados_nuevos if i not in cols_no_graficables]  
    col_para_histograma = encabezados_nuevos
    tam                 = len(col_para_histograma)
    filas               = int(tam / nro_columnas_subplot) if (tam % nro_columnas_subplot) == 0 else int(tam / nro_columnas_subplot) + 1

    fig, ax = plt.subplots(ncols = nro_columnas_subplot, nrows = filas, figsize = figsize_subplots)
    ax = ax.flatten()
    cont = 0
    for i in col_para_histograma:
        #print(i) 
        caja = ax[cont].boxplot(dataframe[i], notch = True, patch_artist = True, vert = False, flierprops = dict(marker='o', alpha = 0.2, markersize = 18, markerfacecolor = 'cornflowerblue',markeredgecolor='none')) #markeredgecolor='white'
        #caja['boxes'].set_facecolor('cornflowerblue')
        #ax[cont].set_xscale('log')
        ax[cont].tick_params(axis='x', labelrotation=90, labelsize=12)
        ax[cont].tick_params(axis='y', labelrotation=90, labelsize=12)
        #ax[cont].set_xlabel(i)
        ax[cont].set_title(i, fontsize = 20)
        cont += 1
    plt.tight_layout();


######################################################################################################################
############################################ FUNCIONES PARA GRAFICAR #################################################
######################################################################################################################

# -------------------------------------------------------------------------
# ------------Para graficar cualquier tipo de serie multivariante ---------
# -------------------------------------------------------------------------

from cycler import cycler

def configurar_parametros_ciclicos(param_color, param_grosor, param_forma):     # Con esta función confiuraremos el tipo de línea, color y grosor para plotear las series de tiempo
    
  if (param_color != False) | (param_grosor != False) | (param_forma != False):
    cantidad = len(param_color)
    ciclador  = []
    for i in range(cantidad):
        parametros = []
        if param_color  != False:
          parametros.append( cycler(color     =  [param_color[i]]      ))       # Acá se configurará el color de cada gráfica
        if param_grosor != False:
          parametros.append( cycler(lw        = [str(param_grosor[i])] ))       # Acá se configurará el ancho de cada gráfica
        if param_forma  != False:
          parametros.append( cycler(linestyle =  [param_forma[i]]      ))       # Acá se configurará la forma de cada gráfica

        for cont, param_i in enumerate(parametros):
          suma_ciclador = (suma_ciclador + param_i) if cont != 0 else param_i
        
        ciclador.append(suma_ciclador)
  else:
    ciclador = False

  return ciclador


def graficador_series(arreglo_de_dataframe, nro_columnas_subplot, figsize_subplot, limites_x, colores_lineas, grosores_lineas, formas_lineas):
    
    nro_de_dataframes   = len(arreglo_de_dataframe)
    arreglo_encabezados = []
    for i in range(nro_de_dataframes):                               # En este ciclo extraemos los encabezados de los dataframes que se plotearan
        encabezados_i = arreglo_de_dataframe[i].columns.tolist()
        arreglo_encabezados.append(encabezados_i)   

    tam   = len(arreglo_encabezados[0])
    filas = int(tam / nro_columnas_subplot) if (tam % nro_columnas_subplot) == 0 else int(tam / nro_columnas_subplot) + 1 
    
    fig, axes = plt.subplots(nrows = filas, ncols = nro_columnas_subplot, dpi = 120, figsize = figsize_subplot)
    #print()
    axes      = axes.flatten() if len(arreglo_de_dataframe[0].columns) > 1 else [axes]
    
    ciclador_configurado = configurar_parametros_ciclicos(param_color = colores_lineas, param_grosor = grosores_lineas, param_forma = formas_lineas)
    for i, ax_i in enumerate(axes):
        
        if i < tam:

            for j in range(nro_de_dataframes):  
                #print(i,j)
                encabezado_j      = arreglo_encabezados[j]
                variable          = encabezado_j[i]
                dataframe_elegido = arreglo_de_dataframe[j]
                datos = dataframe_elegido[variable]

                if ciclador_configurado != False:
                  ax_i.set_prop_cycle(ciclador_configurado[j])                
                ax_i.plot(datos, label = variable)

                if limites_x != False:                                                                 # Este if es para calcular los límites del eje "y" para plotear cuando queremos hacer zoom
                  datos_para_limites = datos[limites_x[0]:limites_x[1]]
                  max_i = max(datos_para_limites) if j == 0 else max(max_i , max(datos_para_limites))
                  min_i = min(datos_para_limites) if j == 0 else min(min_i , min(datos_para_limites))
                  #print('DATAFRAME',j+1,' | VARIABLE',variable,' | ',max_i,min_i)         


        ax_i.legend()
        if limites_x != False:
          ax_i.set(xlim = limites_x)
          ax_i.set_ylim( [min_i,max_i] )
        #ax_i.set_title(variable)
        ax_i.xaxis.set_ticks_position('none')
        ax_i.yaxis.set_ticks_position('none')
        ax_i.spines['right'].set_alpha(0)       # Eliminamos el borde derecho de cada gráfica
        ax_i.spines['top'].set_alpha(0)         # Eliminamos el borde superior de cada gráfica
        ax_i.tick_params(labelsize = 10)        # Reducimos el tamaño de los números de los ejes
        
    plt.tight_layout();                         # Esto hace que los gráficos ocupen mejor el ancho de la pantalla


"""
        if periodicidad_grafica == 'minuto':
           localizador = mdates.MinuteLocator(interval = 15)
           formato = '%H:%M'
        elif periodicidad_grafica == 'hora':
           localizador = mdates.HourLocator()
           formato = '%H:%M'
        elif periodicidad_grafica == 'dia':
           localizador = mdates.DayLocator()
           formato = '%d-%b %Y'
        elif periodicidad_grafica == 'mes':
           localizador = mdates.MonthLocator()
           formato = '%b %Y'
        elif periodicidad_grafica == 'año':
           localizador = mdates.YearLocator()
           formato = '%Y'
        ax_i.xaxis.set_major_locator(localizador)
        ax_i.xaxis.set_major_formatter(mdates.DateFormatter(formato))"""



# -------------------------------------------------------------------------
# ------------- Para descomposición temporal de varias variables ----------
# -------------------------------------------------------------------------

def descomposicion_temporal(dataframe, parametro):
  encabezados = dataframe.columns.tolist()
  df_auxiliar = pd.DataFrame()
  for i in encabezados:
    descomposicion = seasonal_decompose(dataframe[i], extrapolate_trend='freq')
    #print(i, descomposicion, '\n\n')
    df_auxiliar[i] = descomposicion.trend if parametro == 'tendencia' else descomposicion.seasonal if parametro == 'estacional' else descomposicion.resid if parametro == 'residuo' else print('PARÁMETRO ERRÓNEAMENTE ELEGIDO')
  return df_auxiliar


# -------------------------------------------------------------------------
# ----------------- Para graficar los residuos de las series --------------
# -------------------------------------------------------------------------


def graficador_residuos(df_serie_residuo,variable, figsize_subplot):
  fid, axes = plt.subplots(nrows = 3, ncols = 2, dpi = 120, figsize = figsize_subplot)
  axes = axes.flatten()
  encabezados_subplot =  df_serie_residuo.columns.tolist()

  nro_retardos = 40                                                  # Se suele tomar 40
  serie = df_serie_residuo[variable]

  axes[0].plot(serie);axes[0].set_title('Serie '+variable)
  axes[1].hist(serie);axes[1].set_title('Histograma de la Serie '+variable)
  sm.graphics.tsa.plot_acf( serie, ax = axes[2], zero = False, title = 'FACS para los residuos de la serie '+variable)
  sm.graphics.tsa.plot_pacf(serie, ax = axes[3], zero = False, method = ('ols'), title = 'FACP para los residuos la  serie '+variable)
  sm.graphics.tsa.plot_acf( serie**2, ax = axes[4], zero = False, title = 'FACS para los residuos al cuadrado, de la serie '+variable)
  sm.graphics.tsa.plot_pacf(serie**2, ax = axes[5], zero = False, method = ('ols'), title = 'FACP para los residuos al cuadrado, de la serie '+variable)
  plt.tight_layout();               # Esto hace que los gráficos ocupen mejor el ancho de la pantalla


# -------------------------------------------------------------------------
# ---------- Para graficar la Función de Autocorrelación SIMPLE -----------
# -------------------------------------------------------------------------

def graficador_de_FACS(dataframe, nro_columnas_subplot, cols_no_graficables, figsize_subplots, rezagos_a_graficar):
    encabezados         = dataframe.columns.tolist()
    encabezados         = [i for i in encabezados if i not in cols_no_graficables]  
    col_para_histograma = encabezados
    tam                 = len(col_para_histograma)
    filas               = int(tam / nro_columnas_subplot) if (tam % nro_columnas_subplot) == 0 else int(tam / nro_columnas_subplot) + 1

    fig, axes = plt.subplots(ncols = nro_columnas_subplot, nrows = filas, figsize = figsize_subplots)
    axes      = axes.flatten()
    for cont, i in enumerate(col_para_histograma):
      if len(rezagos_a_graficar) > 0:
        sm.graphics.tsa.plot_acf( dataframe[i], ax = axes[cont], zero = False, title = 'FACS para los residuos de la serie '+i, lags = rezagos_a_graficar[cont])
      else:
        sm.graphics.tsa.plot_acf( dataframe[i], ax = axes[cont], zero = False, title = 'FACS para los residuos de la serie '+i)

    plt.tight_layout();


# -------------------------------------------------------------------------
# -- Para graficar la ENVOLVENTE de la Función de Autocorrelación SIMPLE --
# -------------------------------------------------------------------------

def graficador_envolvente_FACS(dataframe, nro_columnas_subplot, cols_no_graficables, figsize_subplots):
    encabezados         = dataframe.columns.tolist()
    encabezados         = [i for i in encabezados if i not in cols_no_graficables]  
    col_para_histograma = encabezados
    tam                 = len(col_para_histograma)
    filas               = int(tam / nro_columnas_subplot) if (tam % nro_columnas_subplot) == 0 else int(tam / nro_columnas_subplot) + 1

    fig, axes = plt.subplots(ncols = nro_columnas_subplot, nrows = filas, figsize = figsize_subplots)
    axes      = axes.flatten()
    for cont, i in enumerate(col_para_histograma):
      pd.plotting.autocorrelation_plot(dataframe[i], ax = axes[cont], label = i)
    plt.tight_layout();


# -------------------------------------------------------------------------
# ----------- Para graficar la Función de Autocorrelación PARCIAL ---------
# -------------------------------------------------------------------------

def graficador_de_FACP(dataframe, nro_columnas_subplot, cols_no_graficables, figsize_subplots, rezagos_a_graficar):
    encabezados         = dataframe.columns.tolist()
    encabezados         = [i for i in encabezados if i not in cols_no_graficables]  
    col_para_histograma = encabezados
    tam                 = len(col_para_histograma)
    filas               = int(tam / nro_columnas_subplot) if (tam % nro_columnas_subplot) == 0 else int(tam / nro_columnas_subplot) + 1

    fig, axes = plt.subplots(ncols = nro_columnas_subplot, nrows = filas, figsize = figsize_subplots)
    axes      = axes.flatten()
    for cont, i in enumerate(col_para_histograma):
      if len(rezagos_a_graficar) > 0:
        sm.graphics.tsa.plot_pacf(dataframe[i], ax = axes[cont], zero = False, method = ('ols'), title = 'FACP para los residuos la  serie '+i, lags = rezagos_a_graficar[cont])
      else:
        sm.graphics.tsa.plot_pacf(dataframe[i], ax = axes[cont], zero = False, method = ('ols'), title = 'FACP para los residuos la  serie '+i) 
    plt.tight_layout();


# -------------------------------------------------------------------------
# ------ Para espectros de frecuencia (FOURIER) de varias variables -------
# -------------------------------------------------------------------------

from re import X
def calcular_fft(serie, frecuencia_muestreo = 1):
    '''y debe ser un vector con números reales
    representando datos de una serie temporal.
    freq_sampleo está seteado para considerar 24 datos por unidad.
    Devuelve dos vectores, uno de frecuencias 
    y otro con la transformada propiamente.
    La transformada contiene los valores complejos
    que se corresponden con respectivas frecuencias.'''
    longitud_serie = len(serie)
    
    frecuencias                  = np.fft.fftfreq(longitud_serie, d = 1/frecuencia_muestreo)
    frecuencias_positivas        = frecuencias[:longitud_serie//2]                            # Esto se hace porque la transformada devuelve potencias en el eje negativo y solo nos interesa la mitad (las positivas) 
    potencia                     = (np.fft.fft(serie)/longitud_serie)
    potencia_para_frec_positivas = potencia[:longitud_serie//2]                               # Esto se hace porque la transformada devuelve potencias en el eje negativo y solo nos interesa la mitad (las positivas)
    return frecuencias_positivas, potencia_para_frec_positivas


def calculador_periodos(arr_picos_frec):
    #print(arr_picos_frec)
    arr_picos_periodos = []
    for i in arr_picos_frec:
        #print(i)
        arr_picos_periodos.append(1/i)
    return arr_picos_periodos


def graficador_espectro(x,y,arr_pico, ax, limites_x, variable, unidad_tiempo):  
  picos_de_tiempo = calculador_periodos(x[arr_pico])
  picos_en_frec   = [recorto_decimales(k) for k in x[arr_pico]]
  picos_de_tiempo = [recorto_decimales(k) for k in picos_de_tiempo]

  ax.scatter(x[arr_pico], y[arr_pico], facecolor = 'r', label = 'Variable ' + str(variable) + '\nLas frencuencias son:' + str(picos_en_frec)+'  ( muestras / '+unidad_tiempo+' )\nY los tiempos son '+str(picos_de_tiempo)+' ('+unidad_tiempo+')')
  ax.plot(x, y)
  ax.set_yscale('log')
  ax.set_xlabel("Frecuencia")
  ax.set_ylabel("Potencia")
  ax.legend(fontsize = 12)

  if (limites_x != False):# & (limites_y != False):
    ax.set(xlim = limites_x)
    #ax.set_ylim( None )
    plt.tight_layout();         # Esto hace que los gráficos ocupen mejor el ancho de la pantalla


import scipy.signal as ss
def espectros(dataframe, muestras_por_unidad_de_tiempo = 1, unidad_de_tiempo = '', nro_columnas_subplot = 1, cols_no_graficables = [], altura_picos = [], nro_picos_a_graficar = [], figsize_subplots = (), limites_eje_x = []):
  encabezados           = dataframe.columns.tolist()
  encabezados           = [i for i in encabezados if i not in cols_no_graficables]
  #print(encabezados)

  nro_series_a_graficar = len(encabezados)
  col_para_histograma = encabezados
  tam                 = len(col_para_histograma)
  filas               = int(tam / nro_columnas_subplot) if (tam % nro_columnas_subplot) == 0 else int(tam / nro_columnas_subplot) + 1
  fig, axis = plt.subplots(nrows = filas, ncols = nro_columnas_subplot, figsize = figsize_subplots) if tam > 1 else plt.subplots(figsize = figsize_subplots)
  axis      = axis.flatten() if tam > 1 else axis

  for cont, var in enumerate(encabezados):

    #print(var, end =' - ')
    serie                  = dataframe[var]
    nro_muestras_totales   = len(serie)
    tiempo_total           = nro_muestras_totales / muestras_por_unidad_de_tiempo



    #tiempo_a_graficar   = tiempo_total                           # [tiempo]
    #tiempo_por_muestra  = tiempo_por_muestra                     # [  tiempo / muestra ]
    #muestras_por_tiempo = 1/tiempo_por_muestra                   # [ muestra / tiempo  ]
    #nro_de_muestras     = tiempo_a_graficar / tiempo_por_muestra # [muestras] = [tiempo] / [tiempo/muestra]
    #frec_muestreo       = nro_de_muestras / tiempo_a_graficar
    frec_muestreo       = muestras_por_unidad_de_tiempo
    #print('La frecuencia de muestreo es:',frec_muestreo, '[ muestras /',unidad_de_tiempo,']')

    frec, pot = calcular_fft(serie, frecuencia_muestreo = frec_muestreo)

    if len(altura_picos) > 0:
      picos = ss.find_peaks(np.abs(pot), prominence = altura_picos[cont])[0]
      #print(picos,'\n',np.abs(pot)[picos])

    if len(nro_picos_a_graficar) > 0:
      picos = encuentro_indices_maximos(np.abs(pot),N = nro_picos_a_graficar[cont]-1)
      frecuencia_minima_a_descartar = 0
      picos = [ k for k in picos if k > frecuencia_minima_a_descartar ]
    #print(picos,'\n\n')

    #n     = 5
    #picos =  encuentro_N_maximos(arreglo = picos, N = n)

    try:
      axis_i = axis[cont]
    except:
      axis_i = axis
    graficador_espectro(x = frec, y = np.abs(pot), arr_pico = picos, ax = axis_i, limites_x = limites_eje_x, variable = var, unidad_tiempo = unidad_de_tiempo)
  

def encuentro_indices_maximos(arreglo, N):
  valores_maximos = encuentro_N_maximos(arreglo, N = N)
  indice          = []
  for cont, i in enumerate(arreglo):
    for j in valores_maximos:
      #print(i,j)
      if i == j:
        indice.append(cont)
  return indice
  

def encuentro_N_maximos(arreglo, N):
    long_arreglo = len(arreglo)
    N            = N if long_arreglo >= N else long_arreglo
    maximos = sorted(arreglo, reverse = True)
    maximos = maximos[:N+1]
    return maximos


def recorto_decimales(num, nro_decimales = 2):
  if num < 1:
    notacion = '.'+str(nro_decimales)+'E'
    num = format(num,notacion)
  else:
    num = int(num*10**nro_decimales)/10**nro_decimales
  return num



######################################################################################################################
###################################### FUNCIONES PARA TESTS EN SERIES DE TIEMPO ######################################
######################################################################################################################


# -------------------------------------------------------------------------
# ------------------- FUNCIONES PARA TEST DE DICKEY-FULLER ----------------
# -------------------------------------------------------------------------


# _________________________________________________________________________
# --------------------(Test completo para varias variables)----------------

def test_dickey_fuller(dataframe):
  encabezados_df = dataframe.columns.tolist()
  for i in encabezados_df:
    resultados = sts.adfuller(dataframe[i])
    print('\nXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX VARIABLE ',i,'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
    print('Ho : La serie NO es estacionaria\nHA : La serie SI es estacionaria')
    print('   Estadístico de prueba = ',resultados[0],'(estad de prueba < valor crítico, entonces se rechaa la hipótesis nula)')
    print('   P-valor               = ',resultados[1],'(         pvalor < alpha        , entonces se rechaa la hipótesis nula)')
    print('   Nro de lags           = ',resultados[2],'(estos son el número de retardos que mediante una regresión, se usaron para calcular el estadístico de prueba, cuando no hay autocorrelaciones, se regresa 0)')
    print('   Nro de observaciones  = ',resultados[3],'(número de observaciones usadas en el análisis)')
    print('alpha_1 =  1%    |    valor crítico_1 = -3.43\nalpha_2 =  5%    |    valor crítico_2 = -2.86\nalpha_3 = 10%    |    valor crítico_3 = -2.56')

# _________________________________________________________________________
# ----------------------(Pvalor para varias variables)---------------------

def pvalor_test_dickey_fuller_multivariante(dataframe):
  encabezados_df      = dataframe.columns.tolist()
  pvalores = []
  for i in encabezados_df:
    pvalor = sts.adfuller(dataframe[i])
    pvalor = recorto_decimales(pvalor[1])
    pvalores.append(pvalor)
  
  max_cifras_variable = max_cantidad_de_caractares_en_listado(listado = encabezados_df)
  max_cifras_pvalores = max_cantidad_de_caractares_en_listado(listado = pvalores)

  for encabezado_i, pvalor_i in zip(encabezados_df, pvalores):
    pvalor_i_imprimir     = impresor_de_caracteres(elemento =     pvalor_i, nro_max_de_caracteres = max_cifras_pvalores, espacio_antes = False)
    encabezado_i_imprimir = impresor_de_caracteres(elemento = encabezado_i, nro_max_de_caracteres = max_cifras_variable)
    print('Se tiene un Pvalor de',pvalor_i_imprimir,' para la variable',encabezado_i_imprimir)


# -------------------------------------------------------------------------
# -------------- FUNCIONES PARA TEST DE CAUSALIDAD DE GRANGER -------------
# -------------------------------------------------------------------------


# _________________________________________________________________________
# --------------------(Test completo para varias variables)----------------

def test_causalidad_de_granger_multivariante(dataframe, resultados_VAR):
  encabezados = dataframe.columns.tolist()
  for i in encabezados:
    var_que_causa = [i]
    vars_causadas = [j for j in encabezados if j not in var_que_causa]
    #print('var_que_causa',var_que_causa,'| vars_causadas',vars_causadas)
    resultado = resultados_VAR.test_causality(vars_causadas, var_que_causa, kind = 'f')   # "f" es la distribución por defecto que viene en la función para el test ("F-test")
    resumen   = resultado.summary()
    print(resumen, '\n')

# Se esperaría un Pvalor pequeño para poder rechazar la Ho (o sea, para poder decir que la variable indicada, SI causa en sentido Granger a las demás)
# Ho: la serie en cuestión NO causa en sentido Granger a las demás

# _________________________________________________________________________
# ----------------------(Pvalor para varias variables)---------------------

def pvalor_test_causalidad_de_granger(dataframe,resultados_VAR):
  encabezados_df      = dataframe.columns.tolist()
  pvalores = []
  for i in encabezados_df:
    var_que_causa = [i]
    vars_causadas = [j for j in encabezados_df if j not in var_que_causa]
    #print('var_que_causa',var_que_causa,'| vars_causadas',vars_causadas)
    resultado     = resultados_VAR.test_causality(vars_causadas, var_que_causa)#, kind = 'f')   # "f" es la distribución por defecto que viene en la función para el test ("F-test")
    pvalor        = recorto_decimales(resultado.pvalue)
    pvalores.append(pvalor)

  max_cifras_variable = max_cantidad_de_caractares_en_listado(listado = encabezados_df)
  max_cifras_pvalores = max_cantidad_de_caractares_en_listado(listado = pvalores)

  for encabezado_i, pvalor_i in zip(encabezados_df, pvalores):
    pvalor_i_imprimir     = impresor_de_caracteres(elemento =     pvalor_i, nro_max_de_caracteres = max_cifras_pvalores, espacio_antes = False)
    encabezado_i_imprimir = impresor_de_caracteres(elemento = encabezado_i, nro_max_de_caracteres = max_cifras_variable)
    print('El Pvalor',pvalor_i_imprimir,'es para indicar si las series son causadas en el sentido Granger por',encabezado_i_imprimir)



######################################################################################################################
################################### FUNCIONES PARA LOS LOGARITMOS EN SERIES DE TIEMPO ################################
######################################################################################################################


# -------------------------------------------------------------------------
# ----------------------- ACÁ SACAMOS LOS LOGARITMOS ----------------------
# -------------------------------------------------------------------------

def calculo_constante_log(serie):
  etiquetas   = serie.columns.tolist()
  colum_serie = len(etiquetas)
  k0 = np.zeros((1,colum_serie))
  
  for iterador_j, encabezado_i in enumerate(etiquetas):
    k0[0,iterador_j] = abs(min(serie[encabezado_i])) + 1   # Acá calculamos los valores constantes que sumaremos a la serie para poder calcular los   max(abs(serie[encabezado_i])) + 1
                                                           # logaritmos (de tal manera que no queden ni ceros, ni números negativos)  
  return k0


def logaritmico(nro_total_de_logaritmos, dataframe):
  encabezados = dataframe.columns.tolist()
  nro_columnas_df = len(dataframe.columns.tolist())
  nro_filas_df    = len(dataframe.index.tolist())
  df_serie_logaritmos_calculados = dataframe.copy()


  k                        = np.zeros((                           1 , nro_columnas_df))
  k_logaritmo              = np.zeros(( (nro_total_de_logaritmos+1) , nro_columnas_df))
  matriz_de_series_de_logs = np.zeros(( (nro_total_de_logaritmos+1) ,    nro_filas_df, nro_columnas_df) ) # Matriz para guardar las conversiones logarítmicas (se una el "+1" porque queremos guardar los valores originales tambn)


  for logaritmo_i in range( 0, (nro_total_de_logaritmos+1) ):                                             # Se una el "+1" porque queremos guardar los valores originales tambn (es decir, los logaritmos y la serie original)

    matriz_de_series_de_logs[logaritmo_i] = df_serie_logaritmos_calculados                                # Esta es la matriz en la que guardaremos las series de tiempo con los logaritmos "n" veces calculados.
    k                                     = calculo_constante_log(serie = df_serie_logaritmos_calculados) # Esta e la constante que se sumará al argumento de los logaritmos para evitar tener log(x) con x <= 0
    k_logaritmo[logaritmo_i]              = k                                                             # Acá guardaremos los valores que se sumaron a los logaritmos "n" veces calculados, para posteriormente revertir la operación "log()"
    serie_log_provisional = []
    for j in range( 0, nro_filas_df ):
      serie_log_provisional.append(  np.log(df_serie_logaritmos_calculados.iloc[j] + k[0])  )  
    
    df_serie_logaritmos_calculados = pd.DataFrame(serie_log_provisional)                                  # Este dataframe se usa solamente para poder ser almacenado en la "matriz de series con logaritmos"
  
  encabezados_log = []
  for i in encabezados:
    encabezados_log.append(str(i)+"_log")

  dataframe_log = pd.DataFrame(matriz_de_series_de_logs[nro_total_de_logaritmos])
  dataframe_log.columns = encabezados_log
  dataframe_log.index   = dataframe.index

  return dataframe_log, k_logaritmo

# -------------------------------------------------------------------------
# --------------------- ACÁ INVERTIMOS LOS LOGARITMOS ---------------------
# -------------------------------------------------------------------------

def des_logaritmico(dataframe, k_logaritmo, nro_total_de_logaritmos, nro_de_datos_conjunto_prueba = False, indices_serie_original = [], nro_diferencias_previo = 0):
  df_sin_log              = pd.DataFrame()
  filas_dataframe_con_log = len(dataframe)
  encabezados             = dataframe.columns.tolist()

  for cont, encabezado_i in enumerate(encabezados):
    
    #print(cont)
    arr_prediccion_sin_log = dataframe[encabezado_i].values
    for fila_i in range(filas_dataframe_con_log):

      #print(fila_i,'\n', arr_prediccion_sin_log)
      valor_sin_log = arr_prediccion_sin_log[fila_i]
      for logaritmo_i in reversed(range(nro_total_de_logaritmos)):

        #print('VALOR ANTES DE LA OPERACIÓN',valor_sin_log)
        valor_sin_log = np.exp(valor_sin_log) - k_logaritmo[logaritmo_i,cont]
      
        #print('ITERACIÓN',cont,'| FILA',fila_i,'| LOGARITMO',logaritmo_i,' | ',valor_sin_log,' k =', k_logaritmo[logaritmo_i,cont],'\nxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n')
      arr_prediccion_sin_log[fila_i] = valor_sin_log
    df_sin_log[str(encabezado_i) + '_sin_log'] = arr_prediccion_sin_log

  if (nro_de_datos_conjunto_prueba != False) & (len(indices_serie_original) >= 0):
    #nro_de_indices_sin_logaritmo = nro_de_indices_sin_logaritmo if nro_de_indices_sin_logaritmo > 0 else len(dataframe)
    nro_de_indices_sin_logaritmo = nro_de_datos_conjunto_prueba + nro_diferencias_previo
    indices_sin_log              = indices_serie_original[-nro_de_indices_sin_logaritmo:]
    df_sin_log.index             = indices_sin_log

  return df_sin_log


######################################################################################################################
################################## FUNCIONES PARA LAS DIFERENCIAS EN SERIES DE TIEMPO ################################
######################################################################################################################


# -------------------------------------------------------------------------
# ---------------------- ACÁ SACAMOS LAS DIFERENCIAS ----------------------
# -------------------------------------------------------------------------

def diferenciador(nro_de_diferencias_total_requeridas = 1, nro_de_datos_conjunto_prueba = 0, dataframe = False):
  nro_total_de_observaciones_ini = len(dataframe)
  serie_diferencias_calculadas   = dataframe.copy()
  column_dataframe               = dataframe.shape[1]
  encabezados_iniciales          = dataframe.columns.tolist()

  nro_total_de_observaciones_dif = nro_total_de_observaciones_ini - nro_de_diferencias_total_requeridas                                       # Tomamos el "largo" de la serie de tiempo multivariante despues de diferenciarse
  limite_division_traint_test    = 0 if nro_de_datos_conjunto_prueba == 0 else nro_total_de_observaciones_dif - nro_de_datos_conjunto_prueba  # Esto se usará para guardar los valores iniciales que se usarán después para REVERTIR las diferencias
  

  valores_iniciales_diferencias = np.zeros((nro_de_diferencias_total_requeridas , column_dataframe))               # En esta matriz guardaremos los valores iniciales que se usarán después para REVERTIR las diferencias
  for diferencia_i in range(0,nro_de_diferencias_total_requeridas):                                                # En este "for" se calculan las diferencias como tal

    #if (limite_division_traint_test != False) & (nro_de_datos_conjunto_prueba != False):
    valores_iniciales_diferencias[diferencia_i] = serie_diferencias_calculadas.iloc[limite_division_traint_test] # NO se usa el "+1" en el indice porque se debe recordar que en python la cuente aranca desde cero, entonces es como si tomaramos la posición anterior a "limite"
    serie_diferencias_calculadas      = serie_diferencias_calculadas.diff().dropna()                       


  encabezados_dif = []                       # Definimos los encabezados de la serie de tiempo diferenciada
  for i in encabezados_iniciales:
    encabezados_dif.append(str(i)+"_dif")


  dataframe_dif = pd.DataFrame(serie_diferencias_calculadas)
  dataframe_dif.columns = encabezados_dif
  dataframe_dif.index   = dataframe.index[nro_de_diferencias_total_requeridas:]

  return dataframe_dif, valores_iniciales_diferencias

# -------------------------------------------------------------------------
# -------------------- ACÁ INVERTIMOS LAS DIFERENCIAS ---------------------
# -------------------------------------------------------------------------

def des_diferenciador(matriz_valores_iniciales_diferencias = False, nro_de_diferencias_total_requeridas = 1, nro_de_datos_conjunto_prueba = 0, indices_serie_original = [], dataframe = False):
  encabezados = dataframe.columns.tolist()
  df_sin_dif  = pd.DataFrame()

  for cont, encabezado_i in enumerate(encabezados):
    
    arr_sin_dif = dataframe[encabezado_i]
    for diferencia_i in reversed(range(nro_de_diferencias_total_requeridas)):

      valor_inicial_a_sumar  = matriz_valores_iniciales_diferencias[diferencia_i, cont]
      data_aun_diferenciada  = arr_sin_dif
      arr_sin_dif            = np.r_[   [valor_inicial_a_sumar], data_aun_diferenciada   ].cumsum()
      #print('VARIABLE',encabezado_i,'| DIFERENCIA',diferencia_i+1,'| valor inicial a sumar',valor_inicial_a_sumar,'\n\n', data_aun_diferenciada,'\n\n',arr_sin_dif,'\nxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n')

    df_sin_dif[encabezado_i+'_sin_dif'] = arr_sin_dif

  if (nro_de_datos_conjunto_prueba  >= 0) & (len(indices_serie_original) >= 0):
    nro_de_datos_conjunto_prueba = nro_de_datos_conjunto_prueba if nro_de_datos_conjunto_prueba > 0 else len(dataframe) - nro_de_datos_conjunto_prueba
    nro_de_fechas_sin_diferencia = nro_de_datos_conjunto_prueba + nro_de_diferencias_total_requeridas
    indices_fechas_sin_dif       = indices_serie_original[-nro_de_fechas_sin_diferencia:]
    df_sin_dif.index             = indices_fechas_sin_dif
  
  return df_sin_dif


######################################################################################################################
#################################### FUNCIONES PARA CALCULAR ERRORES EN REGRESIONES ##################################
######################################################################################################################


# -------------------------------------------------------------------------
# ---------------------- ACÁ SACAMOS LAS DIFERENCIAS ----------------------
# -------------------------------------------------------------------------

def errores_regresiones(lista_yreales, lista_ypredichas, lista_modelos):

    lista_MSE  = []
    lista_RMSE = []
    lista_MAE  = []
    lista_MAPE = []
    lista_R2   = []

    for yreal_i, ypredicha_i in zip(lista_yreales, lista_ypredichas):
      lista_MSE. append(         mean_squared_error            (yreal_i, ypredicha_i)  )
      lista_RMSE.append( np.sqrt(mean_squared_error            (yreal_i, ypredicha_i)) )
      lista_MAE. append(         mean_absolute_error           (yreal_i, ypredicha_i)  )
      lista_MAPE.append(         mean_absolute_percentage_error(yreal_i, ypredicha_i)  )
      lista_R2.  append(         r2_score                      (yreal_i, ypredicha_i)  )

    nro_decimales = 3
    porcentaje    = False
    lista_MSE  = list(map( lambda valor: configurar_nro_decimal(valor, nro_decimales, porcentaje), lista_MSE  ))
    lista_RMSE = list(map( lambda valor: configurar_nro_decimal(valor, nro_decimales, porcentaje), lista_RMSE ))
    lista_MAE  = list(map( lambda valor: configurar_nro_decimal(valor, nro_decimales, porcentaje), lista_MAE  ))
    lista_MAPE = list(map( lambda valor: configurar_nro_decimal(valor, nro_decimales,       True), lista_MAPE ))
    lista_R2   = list(map( lambda valor: configurar_nro_decimal(valor, nro_decimales, porcentaje), lista_R2   ))

    max_cifras_MSE      = max_cantidad_de_caractares_en_listado(listado = lista_MSE  )
    max_cifras_RMSE     = max_cantidad_de_caractares_en_listado(listado = lista_RMSE )
    max_cifras_MAE      = max_cantidad_de_caractares_en_listado(listado = lista_MAE  )
    max_cifras_MAPE     = max_cantidad_de_caractares_en_listado(listado = lista_MAPE )
    max_cifras_R2       = max_cantidad_de_caractares_en_listado(listado = lista_R2   )   
    max_cifras_contador = len(str(len(lista_modelos)))
    max_cifras_modelos  = max_cantidad_de_caractares_en_listado(listado = lista_modelos )  

    print('CANTIDAD TOTAL DE MODELOS = ' + str(len(lista_modelos)) )

    for cont, i in enumerate(lista_modelos):
        
        nro_MSE     = lista_MSE[cont]
        MSE_impr    = impresor_de_caracteres(elemento = nro_MSE,    nro_max_de_caracteres = max_cifras_MSE  )

        nro_RMSE    = lista_RMSE[cont]
        RMSE_impr   = impresor_de_caracteres(elemento = nro_RMSE,   nro_max_de_caracteres = max_cifras_RMSE )

        nro_MAE     = lista_MAE[cont]
        MAE_impr    = impresor_de_caracteres(elemento = nro_MAE,    nro_max_de_caracteres = max_cifras_MAE  )

        nro_MAPE    = lista_MAPE[cont]
        MAPE_impr   = impresor_de_caracteres(elemento = nro_MAPE,   nro_max_de_caracteres = max_cifras_MAPE )

        nro_R2      = lista_R2[cont]
        R2_impr     = impresor_de_caracteres(elemento = nro_R2,     nro_max_de_caracteres = max_cifras_R2 )

        modelo_i    = i
        modelo_impr = impresor_de_caracteres(elemento = modelo_i,   nro_max_de_caracteres = max_cifras_modelos )

        contador_i  = cont + 1
        cont_impr   = impresor_de_caracteres(elemento = contador_i, nro_max_de_caracteres = max_cifras_contador )
    
        print('{}) MSE = {}   |   RMSE = {}   |   MAE = {}   |   MAPE = {} %  |   R2 = {}   |   modelo = {}'.format(cont_impr, MSE_impr, RMSE_impr, MAE_impr, MAPE_impr, R2_impr, modelo_impr))

    print('---------------------------------------------------------------------------------------------------------------------------------------------------------------')


######################################################################################################################
############################ FUNCIÓN PARA GRAFICAR HISTOGRAMAS DE COLOR DE IMÁGENES ##################################
######################################################################################################################



def histogrameador_imagenes (lista_imagenes, nro_columnas_subplot, figsize_subplots, a_BYN = True, con_logaritmo = False):
 
 nro_de_variables       = len(lista_imagenes)

 tam                = nro_de_variables                       # Esta suma es para determinar el número de gráficos que se deben considerar en TOTAL
 filas              = int(tam / nro_columnas_subplot) if (tam % nro_columnas_subplot) == 0 else int(tam / nro_columnas_subplot) + 1  ########################################

 figure, axis = plt.subplots(nrows =filas, ncols = nro_columnas_subplot, figsize = figsize_subplots)

 axis         = axis.flatten() if len(lista_imagenes) > 1 else [axis]

 if a_BYN==False:
    dict_colores  = {'BLUE' : 'cornflowerblue', 'GREEN' : 'lightgreen', 'RED' : 'indianred'}
    dict_grosores = {'BLUE' : 4, 'GREEN' : 4, 'RED' : 4}
 else:
    dict_colores  = {'GRIS' : 'silver'}
    dict_grosores = {'GRIS' :        4}    


 for cont_imagen, imagen_i in enumerate(lista_imagenes):

  imagen_i = np.uint8(imagen_i)

  mascara            = None
  tamanio_histograma = [256]
  rango_de_analisis  = [0, 256]


  for canal_i, nombre_color_i in enumerate(dict_colores.keys()):
    canal_a_analizar = [canal_i]  # es BGR por ende B = 0, G = 1 y R = 2
    histograma_i     = cv2.calcHist([imagen_i], channels = canal_a_analizar, mask = mascara, histSize = tamanio_histograma, ranges = rango_de_analisis)
    histograma_i     = np.log(histograma_i+1) if con_logaritmo == True else histograma_i
    axis[cont_imagen].plot(histograma_i.reshape(1,256)[0], color = dict_colores[nombre_color_i], label = 'imagen_'+str(cont_imagen)+'_'+nombre_color_i, linewidth = dict_grosores[nombre_color_i])
  axis[cont_imagen].legend()  # línea para mostrar los labels en la gráfica

  #axis[cont_imagen].set_title('imagen_'+str(cont_imagen))