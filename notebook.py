#!/usr/bin/env python
# coding: utf-8

# <div class="alert alert-block alert-info">
# 
# Este es un trabajo de análisis de datos de las consolas de videojuegos, su éxito en el mercado de varios países y una predicción de éxito comercial en el futuro.
# 
# </div>

# In[1]:


import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import streamlit as st

import time

from scipy import stats as sps


# In[2]:


df_games_all= pd.read_csv('games.csv')


# In[3]:


df_games_all.info()


# In[4]:


df_games_all['Critic_Score'].unique()


# In[5]:


df_games_all['User_Score'].unique()
print(df_games_all['User_Score'].isna().sum())
df_games_all['User_Score'].unique()


# In[6]:


df_games_all['Rating'].unique()


# In[7]:


df_games_all['Year_of_Release'].unique()


# # Observaciones de los datos
# 
# Después de observar las caracterísitcas de los datos que User_Score es una columna que necesita el cambio de tipo de datos a float64, porque la mayoría tiene números decimales, salvo el texto 'tbd'. En este caso, decidí asignar NaN a 'tbd'. Lo reasignaré al usar pd.to_numeric() con errors='coerce'. Así se tranformará tomáticamente en NaN para permitirme continuar con el análisis de datos sin necesidad de asignar 0 o eliminar la información, lo que alteraría el análisis de los datos de forma negativa porque no son valores en 0 y porque si elimino perdería filas.  
#  
# Critic_Score tiene 8,137 valores no nulos y User_Score tiene 10,014 valores no nulos. No eliminaré los valores nulos porque neceisto analizar la correlación entre reseñas (Critic_Score/User_Score) y deshacerme de los datos ausentes supone perder filas. 
# 
# Después de realizar los cambios, cambiaré los nombres de las columnas a minúsculas, revisaré los datos duplicados y espacios duplicados.

# In[8]:


df_games_all.columns=df_games_all.columns.str.lower()
df_games_all.info()


# In[9]:


df_games_all.columns=df_games_all.columns.str.strip()
df_games_all.head()


# In[10]:


espacios_al_inicio= df_games_all[df_games_all['name'].str.startswith(' ', na=False)]



# In[11]:


espacios_al_final= df_games_all[df_games_all['name'].str.endswith(' ', na=False)]



# In[12]:


df_games_all['name']= df_games_all['name'].str.strip()
espacios_al_inicio= df_games_all[df_games_all['name'].str.startswith(' ', na=False)]


# In[13]:


espacios_al_final= df_games_all[df_games_all['name'].str.endswith(' ', na=False)]


# In[14]:


espacios_inicio_platform= df_games_all[df_games_all['platform'].str.startswith(' ', na=False)]



# In[15]:


espacios_final_platform= df_games_all[df_games_all['platform'].str.endswith(' ', na=False)]


# In[16]:


espacios_inicio_genre= df_games_all[df_games_all['genre'].str.startswith(' ', na=False)]



# In[17]:


espacios_final_genre= df_games_all[df_games_all['genre'].str.endswith(' ', na=False)]


# In[18]:


espacios_inicio_rating= df_games_all[df_games_all['rating'].str.startswith(' ', na=False)]



# In[19]:


espacios_final_rating= df_games_all[df_games_all['rating'].str.endswith(' ', na=False)]


# In[20]:


df_games_all['platform'].unique()


# In[21]:


duplicados_general= df_games_all.duplicated()
duplicados_general.value_counts()


# In[22]:


df_games_all.duplicated().sum()


# In[23]:


df_games_all['name'] = df_games_all['name'].str.lower()

df_games_all['platform'] = df_games_all['platform'].str.lower()

duplicados_name_platform= df_games_all[df_games_all[['name', 'platform', 'year_of_release']].duplicated(keep=False)]

duplicados_name_platform.sort_values(by=['name', 'platform', 'year_of_release'])


# # Observaciones de los duplicados
# 
# Luego de observar el resultado de mi código para encontrar duplicados, llegué a las siguientes conclusiones: 
# 
# 1.-si dos o más filas coinciden en nombre, plataforma y año de lanzamiento, incluso en género, como es el caso de la 604 y la 16230, habrá suficiente evidencia para considerarlas duplicadas. 
# 
# 2.- si no hay nombre qué comparar, es difícil ser categóricos y afirmar que son valores duplicados. Solo coinciden en que son valores ausentes. El caso de las filas 659 y 14244 tienen NaN en nombre y género, pero la plataforma es lo mismo. Esto solo indica que faltan datos para determinar que son verdaderas filas duplicadas porque incluso la fila 14244 tiene valores en 0 en las columnas de las ventas. Por lo tanto, es difícil determinar que se traten de duplicados reales. 
# 
# Con esto en mente, he decido eliminar la fila 16230 porque repiten nombre, año de lanzamiento, plataforma y género. Además, busqué una breve ficha informativa del juego y encontré que el juego madden nfl 13" (ps3, 2012), salió una sola vez en 2012, pero fue adaptado a diferentes plataforma. Por ello, no se debería repetir el nombre de la plataforma en la lista, lo cual ocurre en la fila 16230. 
# 
# Respecto a las fila que tienen datos ausentes, he decidido conservarlas porque no puedo afirmar ni negar que son reales duplicados.

# In[24]:


df_games_all= df_games_all.drop(16230)

duplicados_name_platform= df_games_all[df_games_all[['name', 'platform', 'year_of_release']].duplicated(keep=False)]

duplicados_name_platform.sort_values(by=['name', 'platform', 'year_of_release'])



# Después de eliminar la fila que consideré duplicada, comienzo con la suplantación de los tbd a NaN. Decidí crear una columna llamada score_user_num con el objetivo de conservar los datos originales, pasar la información a datos numéricos y realizar los cambios pertinentes. 

# In[25]:


df_games_all['score_user_num']= pd.to_numeric(df_games_all['user_score'], errors='coerce')
print(df_games_all['score_user_num'].isna().sum())
df_games_all['score_user_num'].unique()


# In[26]:


conteo_tbd = (df_games_all['user_score'] == 'tbd').sum()
conteo_nan_original = df_games_all['user_score'].isna().sum()
conteo_nan_nuevo = df_games_all['score_user_num'].isna().sum()

print("tbd:", conteo_tbd)
print("nan original:", conteo_nan_original)
print("nan nuevo:", conteo_nan_nuevo)
print("suma tbd + nan original:", conteo_tbd + conteo_nan_original)


# # 
# Hipótesis sobre los datos ausentes en critic_score
# 
# La cantidad de valores nulos en Critic_Score varía según la época de lanzamiento del juego, por dos razones principales:
# 
# Juegos de 1980-1999: Presentan entre 90% y 100% de valores nulos porque en esos años no existía una industria consolidada de crítica de videojuegos (plataformas como Metacritic no existían todavía), por lo que la mayoría de estos juegos nunca recibieron una puntuación de crítica.
# Juegos de 2008-2015: Aunque ya existía una industria de crítica más desarrollada, el porcentaje de nulos sigue siendo alto (entre 50% y 62%). Esto probablemente se debe a que cada año se lanzan muchos más juegos, y no todos reciben cobertura de los medios especializados, especialmente los juegos de nicho o de menor presupuesto.
# 

# In[27]:


fil_crict_nul= df_games_all[df_games_all['critic_score'].isna()]



# In[28]:


critic_year_fil= fil_crict_nul['year_of_release'].value_counts()
print(critic_year_fil)


# In[29]:


games_per_year= df_games_all['year_of_release'].value_counts()
print(games_per_year)


# In[30]:


null_games_per_year= critic_year_fil/games_per_year*100
print(null_games_per_year)


# # Hipótesis sobre los datos ausentes en rating
# 
# Respecto a los datos ausentes en rating, encontré que desde 2010 había un alto número de datos ausentes para los videojuegos. Para encontrar la razón, hice una serie de filtros para comprobar que sí hay porcentajes altos de nulos para ciertas plataformas. Por ejemplo: psp tienen 83.4% y psv tiene 65.03% y ps2 tiene 57.7%. En contraste, plataformas como X360, XOne y PC tienen porcentajes bastante más bajos (19-25%). Para hallar alguna explicación entre estos datos, investigué a qué mercados corresponden las plataformas. Lo que tienen en común PSP, PS Vita y PS2 frente a Xbox 360, Xbox One y PC en cuanto a videojuegos es su fuerte identidad enfocada en el mercado japonés. Si un juego nunca se vendió en EE.UU. y Canadá, no habría razón para que la ESRB lo calificara. 
# 
# Para apoyar mi hipótesis, hice una serie de filtros con el objetivo de encontrar los datos nulos de las plataformas con mayor venta en japón:  PSP, PSV, PS2. Después calculé el promedio de ventas para norteamérica y japón. Encontré que en norteamérica se vende menos (44,000 de copias) frente a Japón (76,700 copias). Por lo tanto, los juegos que tienen un mayor mercado en Japón y no llegan norteamérica no tienen rating ESRB.

# In[31]:


fill_rating_nul= df_games_all[df_games_all['rating'].isna()]



# In[32]:


rating_fil= fill_rating_nul['year_of_release'].value_counts()
print(rating_fil)


# In[33]:


rating_per_year= df_games_all['year_of_release'].value_counts()
print(rating_per_year)


# In[34]:


null_rating_per_year= rating_fil/rating_per_year*100
print(null_rating_per_year)


# In[35]:


null_rating_year= df_games_all[(df_games_all['rating'].isna()) & (df_games_all['year_of_release']  >= 2010)]
platform_rating= null_rating_year['platform'].value_counts()
print(platform_rating)


# In[36]:


games_total_year= df_games_all[df_games_all['year_of_release']  >= 2010]
platform_rating_fil= games_total_year['platform'].value_counts()

porcentaje_null_platform= platform_rating/platform_rating_fil*100
print(porcentaje_null_platform)


# In[37]:


nulos_plataformas_jap= fill_rating_nul[fill_rating_nul['platform'].isin(['psp', 'psv', 'ps2'])]
promedio_jp_na= nulos_plataformas_jap[['na_sales', 'jp_sales']].mean()
print(promedio_jp_na)
len(nulos_plataformas_jap)


# # Hipótesis de los datos ausentes en user_score

# 
# Es posible que los datos ausentes en user_score se relacionen con la falta de espacios en internet disponibles para que los usuarios evaluaran los videojuegos. El internet tenía pocos usuarios en la década de 1980, lo que dificulta un registro extenso para los lanzamientos de aquella época. En contraste podría esperarse una mayor cobertura de la valoración de los videojugos por parte de los usuarios en las décadas posteriores.
# 
# Además, como se observó en los datos anteriores, algunas plataformas estaban enfocadas al mercado de Japón. Por ello, los videojuegos solo estaban disponibles para los usuarios de ciertos países que podían evaluarlos y registrar sus opiniones en comunidades regionales.
# 
# Al hacer los filtros para calcular si esta hipótesis era sostenible. Encontré que de 1980 a 1999 los porcentajes van de 92% a 100%. Casi no había forma de que los usuarios calificaran juegos en línea.
# 
# Desde 2000 el porcentaje baja fuerte (69% a 33% en 2003), pero luego sube de nuevo hasta un pico de 62.8% en 2010, y se mantiene alto (44%-57%) en toda la década de 2010-2016. Esto podría indicar que después de 2000 a 2010, hubo una explosión en la cantidad de juegos lanzados, muchos de nicho, que  no acumulan suficientes reseñas de usuarios como para tener un puntaje agregado.
# 
# Respecto a la diferencia de mercados por países, las ventas en Japón mantienen la superioridad con 63 mil copias vendidas frente a norteamérica (con 56 mil). Esta diferencia se presenta en menor medida en comparación con rating. La diferencia geográfica es débil.
# 

# In[38]:


user_score_nul= df_games_all[df_games_all['score_user_num'].isna()]



# In[39]:


user_score_year= user_score_nul['year_of_release'].value_counts()
print(user_score_year) 


# In[40]:


games_per_year= df_games_all['year_of_release'].value_counts()
print(games_per_year)


# In[41]:


null_user_score_xyear= user_score_year/games_per_year*100
print(null_user_score_xyear)


# In[42]:


nulos_plataformas= user_score_nul[user_score_nul['platform'].isin(['psp', 'psv', 'ps2'])]
prom_jp_na= nulos_plataformas[['na_sales', 'jp_sales']].mean()
print(prom_jp_na)
len(nulos_plataformas)


# # ¿Qué hacer con los valores ausentes?
# 
# Los valores ausentes se mantendrán como NaN. Podría trabajar con ellos porque pandas los ignora al momento de hacer cálculos numéricos. 
# 
# No podría reemplazar los valores ausentes con un valor como promedio o 0. Si lo hago alteraría los valores originales para juegos que nunca fueron evaluados o comprados en ciertos países. Estaría inventando valores que no necesariamente reflejan a realidad.
# 
# Una vez explicado lo anterior, calcularé las ventas totales en la columna ventas_totales.

# In[43]:


df_games_all['ventas_totales']= df_games_all[['na_sales', 'eu_sales', 'jp_sales', 'other_sales']].sum(axis=1)
df_games_all[['na_sales', 'eu_sales', 'jp_sales', 'other_sales', 'ventas_totales']].head()


# # Paso 3. Analiza los datos

# In[44]:


games_per_year= df_games_all['year_of_release'].value_counts()
print(games_per_year)


# ## Observaciones de los juegos lanzados en diferentes años
# 
# Después de observar la cantidad de juegos lanzados por año, identifiqué un crecimiento sostenido desde 1991 (con 41 lanzamientos) en adelante. Desde entonces, el número de lanzamientos nunca es inferior a 41. Por otro lado, el período 1980-1990 sería el de años "poco representativos" debido a que hay una baja cantidad de juegos, con fluctuaciones sin una tendencia clara.

# In[45]:


plataforma_ventas_totales=  df_games_all.groupby('platform')['ventas_totales'].sum()

plataforma_ventas_totales= plataforma_ventas_totales.sort_values(ascending=False)
print(plataforma_ventas_totales)


# In[46]:


plataformas_mas_ventas= df_games_all[df_games_all['platform'].isin(['ps2', 'x360', 'ps3', 'wii', 'ds', 'ps'])]


# In[47]:


plataformas_mas_ventas= df_games_all[df_games_all['platform'].isin(['ps2', 'x360', 'ps3', 'wii', 'ds', 'ps'])]

plataformas_ventas_año= plataformas_mas_ventas.groupby(['platform', 'year_of_release'])['ventas_totales'].sum()

tabla_plataformas_año= plataformas_ventas_año.unstack()

print(tabla_plataformas_año)


# In[48]:


fila = tabla_plataformas_año.loc['ds']

años_presentes= fila.index[fila.notnull()]

años_presentes.max()


# In[49]:


def ultimo_año(fila):
    años_presentes = fila.index[fila.notnull()]
    return años_presentes.max()

plataformas_presentes_años=  tabla_plataformas_año.apply(ultimo_año, axis=1)
print(plataformas_presentes_años)


# ## Plataformas que solían ser populares
# 
# El primer paso para encontrar las plataformas que solían ser populares, hice una agrupación por plataforma y año de lanzamiento. Así observé que después de ciertos años, algunas consolas ya no registraban ventas. Depués me dispuse a encontrar el último año en que se registraron ventas, y para conseguirlo escribí una función que me permitiera extraer el año en cuestión. Una vez definida, usé esa función para aplicarla en las demás filas. 
# 
# Los resultados revelaron que  hay 3 plataformas que han dejado de vender hasta el 2016: ds, cuyo último año de ventas llegó a 2013; ps con registros en 2003 y ps2, cuyo último años de ventas fue 2011. 
# 

# In[50]:


años_presentes= fila.index[fila.notnull()]

años_presentes.min()

def primer_año(fila):
    años_presentes = fila.index[fila.notnull()]
    return años_presentes.min()

plataformas_primer_año=  tabla_plataformas_año.apply(primer_año, axis=1)
print(plataformas_primer_año)


# In[51]:


plataformas_añouno_ordenada= plataformas_primer_año.sort_values()

plataformas_añouno_resta= plataformas_añouno_ordenada.diff()

diferencia_años_aparicion= plataformas_añouno_resta.mean()

print(diferencia_años_aparicion)


# In[52]:


duracion_plataforma= plataformas_presentes_años - plataformas_primer_año
duranción_promedio_plataforma= duracion_plataforma.mean()
print(duranción_promedio_plataforma)


# ## ¿Cuánto tardan generalmente las nuevas plataformas en aparecer y las antiguas en desaparecer?
# 
# Después de calcular cada cuántos años tarda en salir una plataforma y cuánto tiempo tienen de vida, encontré que cada 4.2 años aparece una plataforma y suele vivir 13.16 en promedio

# In[53]:


años_representativos= df_games_all[df_games_all['year_of_release'] >2004]
print(años_representativos)


# In[54]:


plataformas_mas_vendidas= años_representativos.groupby('platform')['ventas_totales'].sum()

plataformas_mas_vendidas= plataformas_mas_vendidas.sort_values(ascending=False)
print(plataformas_mas_vendidas)


# In[55]:


evolucion_plataformas= años_representativos.groupby(['platform', 'year_of_release'])['ventas_totales'].sum()

evolucion_plataformas= evolucion_plataformas.unstack()

print(evolucion_plataformas)


# In[56]:


evolucion_plataformas_fil= evolucion_plataformas.loc[evolucion_plataformas.index.isin(['ps2', 'x360', 'ps3', 'wii', 'ds'])]

print(evolucion_plataformas_fil)


# In[57]:


evolucion_plataformas_fil.transpose().plot(kind='line')

plt.xlabel("Años")
plt.ylabel("Ventas")
plt.show()


# ## ¿Qué pataformas registran un declive y cuáles se mantienen a flote?
# 
# Encontré que las pataformas que dominan el mercado de 2016, nacieron 10 años antes de esta fecha, entonces tomé un periodo de análisis de 2005 a 2016. Decidí usar una gráfica de líneas para observar mejor el descenso de cada plataforma y encontré dos grupos: las que van en declive y las que sobreviven.
# 
# Las plataformas que van en descenso son tres: 
# 
# ps2: pico en 2005. Tenía una tendencia a la baja desde el inicio del período. El declive es claro.
# 
# ds: pico en 2007-2008. Entró en una caída sostenida desde 2008 caída, y llega muy baja hacia 2016. Otro ejemplo de un declive claro.
# 
# wii: pico en 2009. La caída abrupta llegó poco después de su pico a finales de 2009. El descenso fue continuo hasta que en en 2014 toco fondo.
# 
# Plataformas con registros de crecimiento:
# 
# ps3 (línea verde) tiene su pico cerca del 2012 y desde ahí su caída es menos abrupta hacia el 2016.
# 
# x360(línea morada) tiene su pico por 2011 y su caída es gradual hacia el 2016.
# 

# In[58]:


evolucion_plat= evolucion_plataformas.loc[evolucion_plataformas.index.isin(['ps4', 'xone', '3ds'])] 

evolucion_plat_potencial= evolucion_plat[[2011.0 , 2012.0, 2013.0, 2014.0, 2015.0, 2016.0]]
print(evolucion_plat_potencial)


# In[59]:


evolucion_plat_potencial.transpose().plot(kind='line')

plt.xlabel("Años")
plt.ylabel("Ventas")
plt.show()


# ## Plataformas potencialmente rentables:
# 
# Usé otra gráfica de líneas para observar cuáles son las plataformas prometedoras. Elegí ps4 y xone porque registran un crecimiento notable desde su año de lanzamiento 2013 hasta el 2016, el único año en el que hay una baja. Por el contrario, 3ds muestra un caso contrario. Desde el año de su lanzamiento va a la baja hasta tocar fondo en 2016.

# In[60]:


años_representativos_plat= años_representativos[años_representativos['platform'].isin(['ps2', 'x360', 'ps3', 'wii', 'ds'])]

años_representativos_plat.boxplot(column='ventas_totales', by= 'platform')
plt.xlabel('Plataformas')
plt.ylabel('Ventas')
plt.title('Ventas globales distribuidas por plataformas (2005-2016)')
plt.show()


# In[61]:


años_representativos_plat= años_representativos[años_representativos['platform'].isin(['ps2', 'x360', 'ps3', 'wii', 'ds'])]

años_representativos_plat.boxplot(column='ventas_totales', by= 'platform')
plt.xlabel('Plataformas')
plt.ylabel('Ventas')
plt.title('Ventas globales distribuidas por plataformas (2005-2016)')
plt.ylim(0, 20)
plt.show()


# ## Observacines de las ventras gobales distribuidas por platafomas
# 
# Después de realizar el diagrama de cajas, donde agrupé por plataformas en un periodo de 2005 a 2016, encontré que las diferencias en las ventas son significativas:
# Wii tiene el outlier máximo absoluto (Wii Sports, ~82.54 millones).
# PS3 y Xbox 360 concentran una franja densa de outliers entre 5 y 20 millones.
# PS2 muestra una acumulación de outliers claramente por debajo de esos 20 millones, es decir, más sesgada hacia valores bajos.
# En el caso de DS,  tiene algunos puntos en el rango mencionado, pero no con la misma concentración. 
# 
# Además, PS3 es la plataforma con mayor RIC (rango intercuartílico), es decir, sus ventas por juego varían más entre el 25% y el 75% central de los datos, comparado con las otras plataformas.
# 
# Respecto a la mediana, en todas las plataformas se aproxima al fondo, se mantiene un nivel muy bajo. Lo anterior indica que hay más juegos acumulados en ventas bajas ) y menos pero más dispersos en la parte alta (ventas más altas dentro de ese rango intercuartílico). Las conclusiones del diagrama de cajas, apoyan una observación que había hecho en celdas anteriores del proyecto: la mayoría de los juegos venden poco, y solo unos pocos venden mucho .

# In[62]:


años_representativos_plat= años_representativos[años_representativos['platform'].isin(['x360'])]
años_representativos_plat.plot.scatter(x= 'critic_score' , y= 'ventas_totales', color='red') 
plt.title('Gráfico de Dispersión: critic_score')
plt.xlabel('Reseñas de la crítica')
plt.ylabel('Ventas totales')
plt.show()


# In[63]:


años_representativos_ps4= años_representativos[años_representativos['platform'].isin(['ps4'])]
años_representativos_ps4.plot.scatter(x= 'critic_score' , y= 'ventas_totales', color='orange') 
print('Correlación:', años_representativos_ps4['critic_score'].corr(años_representativos_ps4['ventas_totales']))

plt.title('Gráfico de Dispersión: critic_score en PS4')
plt.xlabel('Reseñas de la crítica')
plt.ylabel('Ventas totales')
plt.show()


# ## Observaciones del gráfico de dispersión para critic:_score
# 
# La gráfica de distribución para critic_score muestra que hay una correlación positiva entre las ventas altas con las mejores reseñas de la crítica. En este caso, tomé x360 para realizar la evaluación porque fue lanzada en 2005 y se disponen datos sobre su desempeño en años posteriores. 
# 
# Las puntuaciones altas que van del 70 hasta el 100 concentran más ventas. Por otro lado, las puntuaciones más bajas (por debajo de 70) nunca tienen ventas altas.
# 
# En el caso de ps4, hay una correlación positiva, pero moderada (0.40). Los puntos se acumulan del 70 al 80 del eje x (Reseñas de la crítica), con despersiones que alcanzan puntos altos de ventas en el eje y (ventas totales).

# In[64]:

st.subheader('Nombre de las columnas')
st.write(años_representativos_plat.columns)
#st.write(años_representativos_plat['score_user_num'].head())


# In[65]:


años_representativos_plat= años_representativos[años_representativos['platform'].isin(['x360'])]
años_representativos_plat.plot.scatter(x= 'score_user_num' , y= 'ventas_totales', color='blue') 
plt.title('Gráfico de Dispersión: user_score')
plt.xlabel('Reseñas de los jugadores')
plt.ylabel('Ventas totales')
plt.show()


# In[66]:


años_importantes_ps4= años_representativos[años_representativos['platform'].isin(['ps4'])]
años_importantes_ps4.plot.scatter(x= 'score_user_num' , y= 'ventas_totales', color='black') 
print('Correlación:', años_representativos_ps4['score_user_num'].corr(años_representativos_ps4['ventas_totales']))

plt.title('Gráfico de Dispersión: user score en PS4')
plt.xlabel('Reseñas de los jugadores')
plt.ylabel('Ventas totales')
plt.show()


# ## Observaciones para la gráfica de dispersión: user_score
# 
# Para x360, los juegos mejor valorados por los jugadores tienen más ventas, pero hay algunos matices: existen registros de ventas disperas y altas que se concentran en las reseñas de 6-8.
# 
# El caso de ps4 registra una mayor dispersión que x360. La acumulación es notoria en del nivel 7 al 8 en el eje x (Reseñas de los jugadores). Los puntos que llegan más alto en el eje y (Ventas totales) comienzan desde el nivel 5 del eje x, epero registran caídas desde allí en adelante. Luego usar .corr() para obtener la correlación, encontré que prácticamente no la hay porque apenas tiene -0.03.
# 

# In[67]:


años_representativos_plat['name'].head()


# In[68]:


juegos_en_x360 = años_representativos_plat['name'].unique()
print(juegos_en_x360)


# In[69]:


años_representativos_name= años_representativos[años_representativos['name'].isin(juegos_en_x360)]
print(años_representativos_name)


# In[70]:


años_representativos_name[['name', 'platform']].value_counts()


# In[71]:


ventas_grupo= años_representativos_name.groupby(['name', 'platform'])['ventas_totales'].sum().reset_index()
print(ventas_grupo)


# In[72]:

st.subheader('Ventas por videojuegos y plataformas')
ventas_grupo_name= ventas_grupo.pivot(index= 'name', columns= 'platform', values= 'ventas_totales')
st.write(ventas_grupo_name)


# In[73]:


ventas_numero= ventas_grupo_name.count(axis=1)
print(ventas_numero)


# In[74]:


ventas_no_fil= ventas_numero>= 2
ventas_grupo_name[ventas_no_fil]


# In[75]:


ventas_grupo_prom=ventas_grupo_name[ventas_no_fil].mean()
print(ventas_grupo_prom)


# In[76]:


ventas_grupo_prom.min()


# In[77]:


ventas_grupo_prom.max()


# ## Comparación de ventas de los mismos juegos en otras plataformas
# 
# Luego de construir una tabla apartir de la agrupación de nombre, paltaforma y sumar las ventas totales, obtuve una lista que filtré para visualizar los juegos que aparecen en almenos 2 plataformas. Después saqué el promedio de ganancias, y encontré que el mínimo lo registra pc con 247 mil dólares. En cuanto al más alto, lo encontré en ps4, con 1.48 millones de dólares en promedio.
# 
# Con estas magnitudes de ganancias, se esbozan dos grupos: las plataformas que tienen promedios altos: ps4 (1.48), ps3 (0.87), x360 (0.80), xone (0.80).
# Las que tienen promedios bajos: pc (0.25), gba (0.29), 3ds (0.26), por mencionar algunos. 
# 
# Basado en estos hallazgos, observo que los mismos juegos lanzados para varias plataformas no se venden igual porque el comportamiento de ventas suele cambiar de acuerdo a la plataforma. Plataformas como ps4 vende 6 veces más que pc.

# In[78]:


años_representativos.boxplot(column='ventas_totales', by= 'genre')
plt.xlabel('Géneros')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Ventas')
plt.title('Distribución general de los juegos por género(2005-2016)')

plt.show()


# In[79]:


años_representativos.boxplot(column='ventas_totales', by= 'genre')
plt.xlabel('Géneros')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Ventas')
plt.title('Distribución general de los juegos por género(2005-2016)')
plt.ylim(0, 20)
plt.show()


# ## Observaciones sobre la distribución general de los juegos por género
# 
# La mayorías de género tienen ventas bajas. Su cajas están muy cercanas al cero, tienen una altura reducida y su mediana es muy baja. 
# 
# Shooter tiene la mediana y la caja más alta. Los outliers acumulados son superiores a los del resto. 
# 
# El segundo género más vendido es acción, aunque el tamaño de su caja es menor a shooter, también tiene una acumulación de outliers importante.
# 
# Los géneros con ventas más bajas son aventura, pluzzle y strategy. Respecto a los demás géneros (Fighting, Platform, Racing, Role-Playing, Simulation, Sports), mantienen ventas a niveles intermedios.

# In[80]:


plataformas_principales= años_representativos.pivot_table(index= 'platform', values=[ 'na_sales', 'eu_sales', 'jp_sales'], aggfunc= sum)

cinco_plataformas_principaleseu= plataformas_principales.sort_values(by= 'eu_sales' , ascending=False)

cinco_plat_principaleseu= cinco_plataformas_principaleseu['eu_sales'].head(5)
print(cinco_plat_principaleseu)


# In[81]:


total_eusales= plataformas_principales['eu_sales'].sum()
print(total_eusales)


# In[82]:


porcentaje_plataformas_eu= cinco_plat_principaleseu/total_eusales*100
print(porcentaje_plataformas_eu)


# In[83]:


cinco_plataformas_principalesjp= plataformas_principales.sort_values(by= 'jp_sales' , ascending=False)
cinco_plat_principalesjp= cinco_plataformas_principalesjp['jp_sales'].head(5)
print(cinco_plat_principalesjp)


# In[84]:


total_jp_sales= plataformas_principales['jp_sales'].sum()
print(total_jp_sales)


# In[85]:


porcentaje_plataformas_jp= cinco_plat_principalesjp/total_jp_sales*100
print(porcentaje_plataformas_jp)


# In[86]:


cinco_plataformas_principalesna= plataformas_principales.sort_values(by= 'na_sales' , ascending=False)
cinco_plats_principalesna= cinco_plataformas_principalesna['na_sales'].head(5)
print(cinco_plats_principalesna)


# In[87]:


total_na_sales= plataformas_principales['na_sales'].sum()
print(total_na_sales)


# In[88]:


porcentaje_plataformas_na= cinco_plats_principalesna/total_na_sales*100
print(porcentaje_plataformas_na)


# # Paso 4
# 
# ## Observaciones respecto a las variaciones en las cuotas de mercado de una región a otra
# 
# Es notable que x360 tiene una fuerte presencia  en el mercado norteamericano (22.66% de las cuotas ) y en el europeo (16.59). Sin embargo, en esta región ps3, supera a x360 con el  20.23% de cuotas de mercado. 
# 
# Japón registra un caso distinto. La plataforma con mayor porcentaje es ds con 27.83%. La hipótesis que podría explicar lo anterior se fundamente en las restricciones. Aunque esta plataforma está en el mercado mundial, algunos juegos específicos sí se quedaron exclusivamente en Japón y nunca salieron de allí.  
# 
# En segundo lugar aparece 3ds, la plataforma de Japón no es exclusiva de ese país, pero cuenta con un estricto bloqueo regional. Este dato podría explicar por qué tiene un porcentaje alto en comparación con otras regiones en las que ni siquiera aparece en las 5 consolas principales.

# In[89]:


generos_principales= años_representativos.pivot_table(index= 'genre', values=[ 'na_sales', 'eu_sales', 'jp_sales'], aggfunc= sum)

cinco_generos_principaleseu= generos_principales.sort_values(by= 'eu_sales' , ascending=False)

cinco_gen_principaleseu= cinco_generos_principaleseu['eu_sales'].head(5)
print(cinco_gen_principaleseu)


# In[90]:


porcentaje_generos_eu= cinco_gen_principaleseu/total_eusales*100
print(porcentaje_generos_eu)


# In[91]:


cinco_generos_principalesjp= generos_principales.sort_values(by= 'jp_sales' , ascending=False)
cinco_gen_principalesjp= cinco_generos_principalesjp['jp_sales'].head(5)
print(cinco_gen_principalesjp)


# In[92]:


porcentaje_generos_jp= cinco_gen_principalesjp/total_jp_sales*100
print(porcentaje_generos_jp)


# In[93]:


cinco_generos_principalesna= generos_principales.sort_values(by= 'na_sales' , ascending=False)
cinco_gen_principalesna= cinco_generos_principalesna['na_sales'].head(5)
print(cinco_gen_principalesna)


# In[94]:


porcentaje_generos_na= cinco_gen_principalesna/total_na_sales*100
print(porcentaje_generos_na)


# ## Los géneros principales
# 
# El mercado en na y en eu es muy similar. Los primeros 4  género están en el mismo orden aunque con diferentes porcentajes. Para el caso de na, action tiene 22.06% de la cuota de mercado. En cuanto a eu, el mismo género tiene 23.04. La diferencia es poca.
# Sport aparece en segundo lugar para ambos mercados. Na registra 16.45% y eu tiene 16.42%. Aquí la diferencia es mínima. Lo mismo ocurre con shooter: na contabiliza 15.25, y eu, 15.34%. 
# 
# Respecto al mercado japonés, la historia cambia por completo. Role-Playing lidera el mercado con 29.17%. En segundo lugar está action con 17.60%. Una hipótesis que podría explicar lo anterior es la aparición del JRPG (juego de rol japonés), que se diferencia del rol occidental.

# In[95]:


años_representativos.boxplot(column= ['na_sales'], by= 'rating')
plt.xlabel('Rating')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Ventas en Norteamérica')
plt.title('Distribución general de los juegos  en Norteamérica')
plt.ylim(0,5)
plt.show()


# ## ¿Las clasificaciones de ESRB afectan a las ventas en norteamérica?
# 
# Sí, el rating ESRB afecta a las ventas en NA. E (Everyone / Para Todos), E10+ (Everyone 10+ / Para mayores de 10 años), T (Teen) y M (Maduro), concentrar más ventas. Respecto a las clasificaciones más altas, M tienen la caja más grande y E tiene un outlier mayor a 40.
# 
# Por otra parte, las medianas de E, E10+, EC, M y T están más o menos a la misma altura, cerca de cero: esto sugiere que, típicamente, la mayoría de los juegos venden poco en NA sin importar el rating
# 
# 

# In[96]:


años_representativos.boxplot(column= ['eu_sales'], by= 'rating')
plt.xlabel('Rating')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Ventas en Europa')
plt.title('Distribución general de los juegos en Europa')
plt.ylim(0,5)
plt.show()


# ## Observaciones para el mercado en Europa
# 
# M tiene la caja más grande, pero tanto la mediana de esta clasificación como las demás están muy cerca del 0. Otra clasificación que tiene outlier alto es E: está cerca a 30.
# 
# Respecto a los rating que concentran más outliers (hits de ventas) son: E, E10+, M y T. Así que sí, las clasificaciones de ESRB afectan también las ventas en Europa

# In[97]:


años_representativos.boxplot(column= ['jp_sales'], by= 'rating')
plt.xlabel('Rating')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Ventas en Japón')
plt.title('Distribución general de los juegos en Japón')
plt.ylim(0,1)
plt.show()


# ## ¿Las clasificaciones de ESRB afectan a las ventas en norteamérica?
# 
# El rating ESRB sí afecta las ventas en Japón. E, M y T concentrar los outliers aunque es M el que tiene la caja más alta. Respecto a las medianas, son simalres entre sí: todas están cercanas al 0.
# 

# # Paso 5: prueba de las hipótesis

# ### Hipótesis 0: la media de xone es igual a la media de pc. 
# ### Hipótesis 1: la media de xone es diferente a la media de pc.
# 
# El valor alfa se establece en 0.05 porque es un margen de error tolerable en caso rechazar la hipótesis 0 cuando en realidad era verdadera.

# In[98]:


#H0: μ(xone) = μ(pc)
#H1: μ(xone) ≠ μ(pc)

alpha = 0.05

extraccion_xone= años_representativos[años_representativos['platform']=='xone']['score_user_num']

extaccion_pc= años_representativos[años_representativos['platform']=='pc']['score_user_num']

hip_xone_pc = sps.ttest_ind(extraccion_xone, extaccion_pc, equal_var=False)

p_valor = hip_xone_pc.pvalue 

if p_valor  < alpha:
    print('Rechazamos la hipótesis nula (H0): Existe una diferencia en las calificaciones promedio de los usuarios para las plataformas Xbox One y PC')
else:
    print('Se mantiene la hipótesis nula (H0): No hay evidencia de diferencia significativa.')


# ### H0: la media de las calificaciones para el género acción es igual a la media de las calificaciones para el género de deportes.
# 
# ### H1: la media de las calificaciones para el género acción es diferente a la media de las calificaciones para el género de deportes.
# 
# El valor alfa está establecido en 0.05 en caso de rechazar la hipótesis nula cuando en realidad era verdadera.

# In[99]:


#H0: μ(action) = μ(sports)
#H1: μ(action) ≠ μ(sports)

alpha = 0.05

extraccion_action= años_representativos[años_representativos['genre']=='Action']['score_user_num']

extraccion_sports= años_representativos[años_representativos['genre']=='Sports']['score_user_num']

hip_action_sports = sps.ttest_ind(extraccion_action, extraccion_sports, equal_var=False)

p_valor = hip_action_sports.pvalue 

if p_valor  < alpha:
    print('Rechazamos la hipótesis nula (H0): Existe una diferencia en las calificaciones promedio de los usuarios para los géneros de Acción y Deportes.')
else:
    print('Se mantiene la hipótesis nula (H0): No hay evidencia de diferencia significativa.')


# In[100]:


print(años_representativos["genre"].unique())


# # Conclusión general
# 
# Después de analizar los datos de las plataformas, tendencias de ventas en videojuegos, popularidad de plataformas y revisión de hipótesis, concluyo lo siguiente:
# 
# ## ¿Cómo evoluciona el mercado y qué plataformas son rentables? 
# 
# Ciclo de vida y aparición: El mercado muestra un crecimiento sostenido desde 1991 (descartando el periodo 1980–1990 por baja representatividad). Las consolas surgen en promedio cada 4.2 años y tienen una vida útil estimada de 13.16 años.
# 
# Plataformas a evitar vs. priorizar: Se identificó que consolas históricas como PS2, DS y Wii se encuentran en claro declive o muertas comercialmente (DS dejó de vender en 2013, PS2 en 2011). Para focalizar esfuerzos comerciales, las plataformas clave con potencial de rentabilidad al cierre del periodo analizado son PS4 y Xbox One, gracias a su fuerte curva de crecimiento desde su lanzamiento en 2013.
# 
# Efecto multiplataforma: Lanzar un mismo juego en múltiples consolas no garantiza el mismo rendimiento. PS4 encabeza los promedios de ganancias por juego multiversión (1.48 millones de dólares), superando hasta por 6 veces el promedio de plataformas como PC (0.247 millones).
# 
# ## ¿Qué factores determinan el éxito comercial de un juego (Géneros, Reseñas y ESRB)? 
# 
# La regla general de ventas: La gran mayoría de los títulos registran ventas bajas (con medianas cercanas a cero), mientras que el volumen fuerte de ingresos se concentra en unos pocos éxitos masivos (outliers, como Wii Sports).
# Géneros dominantes: Shooter y Action lideran el mercado global en términos de cajas de distribución altas, densidad de éxitos y rentabilidad. En contraparte, Adventure, Puzzle y Strategy registran los desempeños más bajos.
# 
# Impacto de las reseñas: Existe una correlación positiva directa. Las puntuaciones altas de la crítica (critic_score de 70 a 100) y de los usuarios (user_score de 6 a 8) son filtros críticos: ningún juego con reseñas bajas alcanza ventas altas.
# 
# Clasificación ESRB: Las calificaciones (especialmente M, E, E10+ y T) concentran los principales picos de ventas (hits) en los mercados occidentales y Japón, aunque el comportamiento base de la industria mantiene medianas de ventas bajas sin importar el público objetivo.
# 
# ## ¿Cómo se comportan los mercados regionales? 
# 
# Occidente (Norteamérica y Europa): Son mercados altamente simétricos donde las preferencias se inclinan hacia los géneros Action, Sports y Shooter. En cuanto a cuotas de mercado, Xbox destaca con fuerza en Norteamérica, mientras que PlayStation lidera en Europa.
# 
# Japón (Un mercado único): Presenta un comportamiento totalmente disonante debido a factores de exclusividad y bloqueo regional, donde las consolas portátiles (DS y 3DS) dominan las listas y el género Role-Playing (JRPG) lidera indiscutiblemente las preferencias locales por encima de la acción.
# 
# ## Hallazgos Estadísticos y Pruebas de Hipótesis 
# 
# Las evaluaciones estadísticas validaron que:
# Xbox One vs. PC: Se retiene la hipótesis nula.  Lo anterior confirma que no existe una diferencia estadísticamente significativa en las calificaciones promedio que otorgan los usuarios entre ambas plataformas.
# 
# Acción vs. Deportes: Se retiene la hipótesis nula. Las calificaciones promedio de los usuarios entre ambos géneros son estadísticamente equivalentes.

menu= st.selectbox('Elige las categorías de las plataformas', ['Todas', 'En declive', 'Estables', 'Rentables'])

evolucion_todas_plataformas = evolucion_plataformas.loc[evolucion_plataformas.index.isin(['ps4','xone','3ds','ps2','x360','ps3','wii','ds'])]

if menu == 'Todas' :
    plata_filt= evolucion_plataformas.loc[evolucion_plataformas.index.isin(['ps4','xone','3ds','ps2','x360','ps3','wii','ds'])]
    color = 'yellow'
elif menu == 'En declive':
    plata_filt = evolucion_plataformas.loc[evolucion_plataformas.index.isin(['ps2','ds','wii'])]
    color = 'red'
elif menu == 'Estables': 
    plata_filt = evolucion_plataformas.loc[evolucion_plataformas.index.isin(['ps3', 'x360'])]
    color = 'green'
elif menu == 'Rentables':
   plata_filt = evolucion_plataformas.loc[evolucion_plataformas.index.isin(['ps4', 'xone'])]
   color = 'purple'

grafico = st.empty()

years= [2005.0,  2006.0,  2007.0,  2008.0,  2009.0,  2010.0,  2011.0, 2012.0,  2013.0,  2014.0,  2015.0,  2016.0]

for year in years:
    fig, ax = plt.subplots()
    ax.barh(plata_filt.index, plata_filt[year], color=color)
    ax.set_title(f' Ventas de {year}')
    ax.set_xlabel('Ventas')
    ax.set_ylabel('Consolas')
    grafico.pyplot(fig)
    time.sleep(1.5)
