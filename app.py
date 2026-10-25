from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title='Observatorio de Turismo Sostenible',layout='wide',initial_sidebar_state='expanded')
BASE=Path(__file__).resolve().parent
FOREST='#264D3C'; SAGE='#91A88E'; SAND='#E8DCC6'; GOLD='#D5AD55'; WHITE='#FFFFFF'
st.markdown('''<style>
:root{color-scheme:light!important}html,body,[data-testid="stApp"],.stApp,[data-testid="stMain"],.main{background:#FFFFFF!important;color:#264D3C!important}
[data-testid="stHeader"]{background:#FFFFFF!important}
[data-testid="stSidebar"],[data-testid="stSidebar"]>div:first-child{background:#264D3C!important}
[data-testid="stSidebar"] *{color:#FFFFFF!important}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p{color:#FFFFFF!important}
[data-testid="stSidebar"] [data-testid="stRadio"] label{color:#FFFFFF!important}
h1,h2,h3,h4,p,label,small,[data-testid="stCaptionContainer"], [data-testid="stMetricLabel"], [data-testid="stMetricValue"]{color:#264D3C!important}
h1{font-size:2rem!important;font-weight:750!important}h2,h3{font-weight:700!important}
[data-testid="stMetric"]{background:#FFFFFF;border:1px solid #E8DCC6;border-radius:14px;padding:16px;box-shadow:0 2px 9px #264D3C0D}
[data-testid="stMetricValue"]{font-weight:750!important}
[data-testid="stPlotlyChart"]{background:#FFFFFF;border:1px solid #E8DCC6;border-radius:14px;padding:8px}
[data-testid="stWidgetLabel"] p{color:#264D3C!important;font-weight:600!important}
[data-baseweb="select"]>div,[data-baseweb="input"]>div,input,[data-testid="stDateInput"]>div>div{background:#FFFFFF!important;color:#264D3C!important;border-color:#91A88E!important}
[data-baseweb="select"] span,[data-baseweb="select"] input,input{color:#264D3C!important}
[data-testid="stSidebar"] button{background:#FFFFFF!important;color:#264D3C!important;border-color:#D5AD55!important}
[data-testid="stSidebar"] button *{color:#264D3C!important}
.stButton>button[kind="primary"], [data-testid="stDownloadButton"] button{background:#264D3C!important;color:white!important;border:none!important;border-radius:9px}
[data-testid="stDownloadButton"] button *{color:white!important}
[data-testid="stAlert"]{border-radius:10px}
hr{border-color:#E8DCC6!important}
</style>''',unsafe_allow_html=True)

@st.cache_data
def load(name):
    d=pd.read_csv(BASE/name)
    d['Fecha']=pd.to_datetime(d['Fecha'],errors='coerce')
    d['Viajero']=pd.to_numeric(d['Viajero'],errors='coerce')
    return d.dropna(subset=['Fecha','Viajero']).sort_values('Fecha')

total=load('serie_total_mensual.csv')
air=pd.concat([load('train_via_aerea_mensual.csv'),load('test_via_aerea_mensual.csv')]).drop_duplicates('Fecha',keep='last')
joined=total.merge(air,on='Fecha',how='inner',suffixes=('_total','_aerea')).sort_values('Fecha')

MESES=['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic']
COLS={'Total de viajeros':'Viajero_total','Vía aérea':'Viajero_aerea'}
SERIES_NOMBRE={'Total':'Total','Via_Aérea':'Vía aérea','Via_Terrestre':'Vía terrestre','Via_Marítima':'Vía marítima','Reg_América Del Centro':'Región Centroamérica','Reg_América Del Norte':'Región Norteamérica','Reg_Europa':'Región Europa'}
CLUSTER={'Total':0,'Via_Aérea':0,'Via_Terrestre':0,'Via_Marítima':1,'Reg_América Del Centro':0,'Reg_América Del Norte':2,'Reg_Europa':2}
DIST={'Via_Marítima':8.66,'Reg_América Del Norte':7.42,'Reg_Europa':6.96,'Via_Terrestre':6.57,'Via_Aérea':6.52,'Reg_América Del Centro':6.49,'Total':6.21}

def style(fig,h=340,**xaxis):
    fig.update_layout(height=h,margin=dict(l=70,r=15,t=15,b=70),paper_bgcolor=WHITE,plot_bgcolor=WHITE,font=dict(color=FOREST,size=12),
                      legend=dict(orientation='h',y=1.14,x=0),xaxis=dict(automargin=True,showgrid=False,**xaxis),yaxis=dict(automargin=True,gridcolor=SAND,zerolinecolor=SAND))
    return fig

with st.sidebar:
    st.markdown('## Observatorio')
    st.caption('TURISMO SOSTENIBLE')
    section=st.radio('Navegación',['Inicio','Evolución','Vías de ingreso','Predicciones','Metodología'],label_visibility='collapsed')
    st.divider()
    st.markdown('**Base de migración · 2009–2026**')
    st.caption('Diseño inspirado en las referencias del equipo. Datos reales suministrados para el laboratorio.')

st.title('Observatorio de Turismo Sostenible')
st.caption('Análisis interactivo de viajeros · Base de migración 2009–junio 2026')

if section=='Predicciones':
    st.subheader('Resultados de modelos predictivos')
    st.caption('Métricas de los Laboratorios 1 (series de tiempo clásicas) y 2 (LSTM) sobre el mismo conjunto de prueba: abril 2021 a junio 2026. No se entrenaron modelos nuevos. RMSE y MAE miden el error promedio en viajeros: menor es mejor.')
    met=pd.read_csv(BASE/'metricas_modelos.csv').dropna(subset=['RMSE'])
    met['Serie']=met.Serie.map(SERIES_NOMBRE)
    c1,c2=st.columns([1.3,1])
    with c1: serie_m=st.selectbox('Serie a evaluar',list(SERIES_NOMBRE.values()))
    with c2: error=st.radio('Métrica de error',['RMSE','MAE'],horizontal=True)
    sel=met[met.Serie==serie_m].sort_values(error,ascending=False)
    best=sel.iloc[-1]
    st.success(f'Menor {error} en {serie_m}: {best.Modelo} ({best.Laboratorio}) con {best[error]:,.0f} viajeros de error.')
    st.markdown('#### Error por modelo')
    st.caption('Cada barra es un modelo evaluado en la serie elegida. Verde: Laboratorio 1, dorado: Laboratorio 2 (LSTM).')
    fig=px.bar(sel,x=error,y='Modelo',color='Laboratorio',orientation='h',color_discrete_map={'Lab1':FOREST,'Lab2':GOLD})
    fig.update_traces(hovertemplate='%{y}<br>'+error+': %{x:,.0f}<extra></extra>')
    st.plotly_chart(style(fig,330),width='stretch',theme=None)
    g1,g2=st.columns(2)
    with g1:
        st.markdown('#### Mejor modelo del Lab 1 frente al mejor LSTM')
        st.caption('Solo las series total y vía aérea tienen LSTM. En total gana el LSTM; en vía aérea gana el modelo clásico.')
        cmp=met[met.Serie.isin(['Total','Vía aérea'])].copy()
        cmp['Grupo']=cmp.Laboratorio.map({'Lab1':'Mejor del Lab 1','Lab2':'Mejor LSTM'})
        cmp=cmp.loc[cmp.groupby(['Serie','Grupo']).RMSE.idxmin()]
        f=px.bar(cmp,x='Serie',y='RMSE',color='Grupo',barmode='group',hover_data=['Modelo'],color_discrete_map={'Mejor del Lab 1':FOREST,'Mejor LSTM':GOLD})
        st.plotly_chart(style(f,320),width='stretch',theme=None)
    with g2:
        st.markdown('#### Series más y menos parecidas entre sí')
        st.caption('Distancia media de cada serie al resto según sus características (catch22, Lab 2). Mayor distancia indica un comportamiento más atípico. El color es el grupo de KMeans.')
        d=pd.DataFrame({'Serie':[SERIES_NOMBRE[k] for k in DIST],'Distancia media':list(DIST.values()),'Grupo':[f'Grupo {CLUSTER[k]+1}' for k in DIST]})
        f=px.bar(d,x='Distancia media',y='Serie',color='Grupo',orientation='h',color_discrete_sequence=[FOREST,GOLD,SAGE])
        st.plotly_chart(style(f,320).update_yaxes(autorange='reversed'),width='stretch',theme=None)
    with st.expander('Tabla comparativa de todos los modelos'):
        st.dataframe(met.sort_values(['Serie','RMSE'])[['Serie','Laboratorio','Modelo','MAE','RMSE']],width='stretch',hide_index=True)
    st.info('Limitaciones: el conjunto de prueba incluye la recuperación posterior a 2020, por lo que los errores son altos y difíciles de generalizar. El SARIMA de vía aérea no produjo un pronóstico válido y no aparece. Las métricas se tomaron de los cuadernos originales, no se recalcularon.')
    st.stop()
if section=='Metodología':
    st.subheader('Fuentes y limitaciones')
    st.write('Datos: serie_total_mensual.csv y archivos train/test_via_aerea_mensual.csv. Se emparejan fechas coincidentes. La serie total procesada contiene valores decimales; no se deben interpretar automáticamente como conteos oficiales exactos.')
    st.write('Las selecciones de períodos se comparten entre visualizaciones. Las comparaciones describen patrones, no demuestran relaciones causales.')
    st.stop()

st.markdown('#### Filtros de exploración')
f1,f2,f3,f4=st.columns([1.2,1.2,1.5,1.3])
minyear=int(joined.Fecha.dt.year.min()); maxyear=int(joined.Fecha.dt.year.max())
with f1: year_from=st.selectbox('Desde el año',list(range(minyear,maxyear+1)),index=0)
with f2: year_to=st.selectbox('Hasta el año',list(range(maxyear, minyear-1,-1)),index=0)
with f3: level=st.selectbox('Nivel de detalle',['Mensual','Anual'])
with f4: series=st.selectbox('Serie destacada',['Total de viajeros','Vía aérea'])
search=st.text_input('Buscar período (opcional)',placeholder='Ejemplo: 2022 o 2022-05',help='Filtra los períodos visibles por año o año-mes. Deja vacío para mostrar todo.')
if year_from>year_to:
    st.warning('El año inicial debe ser menor o igual al final.');st.stop()
subset=joined[joined.Fecha.dt.year.between(year_from,year_to)].copy()
if search.strip():
    subset=subset[subset.Fecha.dt.strftime('%Y-%m').str.contains(search.strip(),regex=False)]
if subset.empty:
    st.info('No hay datos para la selección. Modifica los filtros.');st.stop()

m1,m2,m3=st.columns(3)
m1.metric('Total de viajeros (serie)',f'{subset.Viajero_total.sum():,.0f}')
m2.metric('Promedio mensual',f'{subset.Viajero_total.mean():,.0f}')
m3.metric('Viajeros por vía aérea',f'{subset.Viajero_aerea.sum():,.0f}')

col=COLS[series]
if section=='Evolución':
    st.subheader(f'Patrones de {series.lower()}')
    st.caption('Hallazgos descriptivos: muestran cuándo se concentra la actividad, no por qué.')
    base=joined[joined.Fecha.dt.year.between(year_from,year_to)].copy()
    base['Año']=base.Fecha.dt.year; base['Mes']=base.Fecha.dt.month
    h1,h2=st.columns(2)
    with h1:
        st.markdown('#### Calendario de viajeros')
        st.caption('Cada celda es un mes. Más oscuro, más viajeros.')
        piv=base.pivot_table(index='Año',columns='Mes',values=col,aggfunc='sum').reindex(columns=range(1,13))
        f=px.imshow(piv.values,x=MESES,y=piv.index.astype(str),color_continuous_scale=[WHITE,SAGE,FOREST],aspect='auto',labels=dict(color='Viajeros'))
        st.plotly_chart(style(f,380),width='stretch',theme=None)
    with h2:
        st.markdown('#### Estacionalidad por mes')
        st.caption('Distribución de cada mes en los años seleccionados. La línea central es la mediana.')
        base['Nombre']=base.Mes.map(lambda m:MESES[m-1])
        f=px.box(base,x='Nombre',y=col,category_orders={'Nombre':MESES},color_discrete_sequence=[FOREST],labels={col:'Viajeros','Nombre':'Mes'})
        st.plotly_chart(style(f,380),width='stretch',theme=None)
    st.markdown('#### Variación frente al mismo período del año anterior')
    st.caption('Verde: crece. Dorado: cae. Un año parcial (2026) solo incluye los meses disponibles.')
    yo=joined.set_index('Fecha')[col].pct_change(12).mul(100).rename('Variación').reset_index().dropna()
    yo=yo[yo.Fecha.dt.year.between(year_from,year_to)]
    if level=='Anual': yo=yo.groupby(yo.Fecha.dt.year.astype(str))['Variación'].mean().reset_index().rename(columns={'Fecha':'Período'})
    else: yo['Período']=yo.Fecha.dt.strftime('%Y-%m')
    yo['Dirección']=yo.Variación.ge(0).map({True:'Crece',False:'Cae'})
    f=px.bar(yo,x='Período',y='Variación',color='Dirección',color_discrete_map={'Crece':FOREST,'Cae':GOLD},labels={'Variación':'Variación (%)'})
    f.update_xaxes(type='category',categoryorder='category ascending',nticks=14,tickangle=-35)
    st.plotly_chart(style(f,330),width='stretch',theme=None)
    st.stop()
if section=='Vías de ingreso':
    st.subheader('Vía aérea frente al total de viajeros')
    st.caption('La serie de vía aérea es una parte del total. La comparación describe cómo se mueven juntas, no demuestra causas.')
    b=subset.copy()
    if level=='Anual':
        b['Período']=b.Fecha.dt.year.astype(str); b=b.groupby('Período',as_index=False)[['Viajero_total','Viajero_aerea']].sum()
    else: b['Período']=b.Fecha.dt.strftime('%Y-%m')
    b['Participación aérea (%)']=b.Viajero_aerea/b.Viajero_total*100
    st.markdown('#### Participación de la vía aérea en el total')
    st.caption('Porcentaje del total que corresponde a vía aérea.')
    f=px.area(b,x='Período',y='Participación aérea (%)',color_discrete_sequence=[GOLD])
    f.update_xaxes(type='category',nticks=14,tickangle=-35)
    st.plotly_chart(style(f,320),width='stretch',theme=None)
    st.markdown('#### Total frente a vía aérea')
    st.caption('Cada punto es un período. Si los puntos suben juntos, ambas series se mueven en la misma dirección.')
    f=px.scatter(b,x='Viajero_total',y='Viajero_aerea',hover_name='Período',color_discrete_sequence=[FOREST],labels={'Viajero_total':'Total de viajeros','Viajero_aerea':'Vía aérea'})
    f.update_traces(marker=dict(size=9,opacity=.75,line=dict(color=WHITE,width=1)))
    st.plotly_chart(style(f,340),width='stretch',theme=None)
    st.stop()

st.subheader('Evolución de viajeros a través del tiempo')
st.caption('Selecciona puntos en la gráfica para actualizar la comparación inferior. También puedes usar zoom y consultar los valores al pasar el cursor.')
plot=subset.copy()
if level=='Anual':
    plot['Periodo']=plot.Fecha.dt.year.astype(str)
    plot=plot.groupby('Periodo',as_index=False)[['Viajero_total','Viajero_aerea']].sum()
else:
    plot['Periodo']=plot.Fecha.dt.strftime('%Y-%m')
plot['Total de viajeros']=plot.Viajero_total
plot['Vía aérea']=plot.Viajero_aerea
selected_name=series
fig=px.line(plot,x='Periodo',y=['Total de viajeros','Vía aérea'],markers=True,
            color_discrete_map={'Total de viajeros':FOREST,'Vía aérea':GOLD},labels={'value':'Viajeros','variable':'Serie'})
fig.update_traces(marker_size=7,selected=dict(marker=dict(size=12)),unselected=dict(marker=dict(opacity=.4)))
fig.update_layout(dragmode='select',hovermode='x unified',height=390,margin=dict(l=70,r=15,t=20,b=70),
                  paper_bgcolor=WHITE,plot_bgcolor=WHITE,font=dict(color=FOREST,size=12),
                  legend=dict(orientation='h',y=1.13,x=0),
                  xaxis=dict(automargin=True,title='Período',type='category',showgrid=False,tickangle=-35,nticks=14),
                  yaxis=dict(automargin=True,title='Viajeros',gridcolor=SAND,zerolinecolor=SAND))
fig.update_traces(line=dict(width=3))

event=st.plotly_chart(fig,width='stretch',on_select='rerun',selection_mode=('points','box','lasso'),key='time_selection',theme=None)
chosen=[]
try:
    for p in event.selection.points:
        idx=p.get('point_index',p.get('point_number'))
        if idx is not None and 0<=int(idx)<len(plot): chosen.append(str(plot.iloc[int(idx)].Periodo))
except (AttributeError,TypeError,ValueError):
    pass
chosen=list(dict.fromkeys(chosen))
if chosen:
    detail=plot[plot.Periodo.isin(chosen)].copy()
    st.success(f'{len(chosen)} período(s) seleccionado(s). La gráfica inferior muestra únicamente esa selección.')
else:
    detail=plot.copy()

st.subheader('Comparación de viajeros totales y vía aérea')
st.caption('Las barras cambian al seleccionar períodos en la gráfica superior. Usa el filtro de detalle para comparar por mes o año.')
long=detail.melt(id_vars='Periodo',value_vars=['Total de viajeros','Vía aérea'],var_name='Serie',value_name='Viajeros')
fig2=px.bar(long,x='Periodo',y='Viajeros',color='Serie',barmode='group',
            color_discrete_map={'Total de viajeros':FOREST,'Vía aérea':GOLD},
            labels={'Periodo':'Período'})
fig2.update_layout(height=355,margin=dict(l=70,r=15,t=15,b=70),paper_bgcolor=WHITE,plot_bgcolor=WHITE,
                   font=dict(color=FOREST,size=12),legend=dict(orientation='h',y=1.14,x=0),
                   xaxis=dict(automargin=True,type='category',tickangle=-35,nticks=15,showgrid=False),
                   yaxis=dict(automargin=True,gridcolor=SAND,zerolinecolor=SAND))
fig2.update_traces(hovertemplate='%{x}<br>%{fullData.name}: %{y:,.0f}<extra></extra>')
st.plotly_chart(fig2,width='stretch',theme=None)

with st.expander('Ver y descargar los datos utilizados'):
    st.dataframe(detail[['Periodo','Total de viajeros','Vía aérea']],width='stretch',hide_index=True)
    st.download_button('Descargar datos filtrados (CSV)',detail.to_csv(index=False).encode('utf-8-sig'),'datos_filtrados.csv','text/csv')
