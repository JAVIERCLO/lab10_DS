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
    st.info('Esta sección se completará al incorporar los resultados de evaluación de los tres modelos de los laboratorios 1 y/o 2. No se entrenan modelos nuevos.')
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
fig.update_layout(dragmode='select',hovermode='x unified',height=390,margin=dict(l=15,r=15,t=20,b=35),
                  paper_bgcolor=WHITE,plot_bgcolor=WHITE,font=dict(color=FOREST,size=12),
                  legend=dict(orientation='h',y=1.13,x=0),
                  xaxis=dict(title='Período',type='category',showgrid=False,tickangle=-35,nticks=14),
                  yaxis=dict(title='Viajeros',gridcolor=SAND,zerolinecolor=SAND))
fig.update_traces(line=dict(width=3))

event=st.plotly_chart(fig,use_container_width=True,on_select='rerun',selection_mode=('points','box','lasso'),key='time_selection')
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
fig2.update_layout(height=355,margin=dict(l=15,r=15,t=15,b=35),paper_bgcolor=WHITE,plot_bgcolor=WHITE,
                   font=dict(color=FOREST,size=12),legend=dict(orientation='h',y=1.14,x=0),
                   xaxis=dict(type='category',tickangle=-35,nticks=15,showgrid=False),
                   yaxis=dict(gridcolor=SAND,zerolinecolor=SAND))
fig2.update_traces(hovertemplate='%{x}<br>%{fullData.name}: %{y:,.0f}<extra></extra>')
st.plotly_chart(fig2,use_container_width=True)

with st.expander('Ver y descargar los datos utilizados'):
    st.dataframe(detail[['Periodo','Total de viajeros','Vía aérea']],use_container_width=True,hide_index=True)
    st.download_button('Descargar datos filtrados (CSV)',detail.to_csv(index=False).encode('utf-8-sig'),'datos_filtrados.csv','text/csv')
