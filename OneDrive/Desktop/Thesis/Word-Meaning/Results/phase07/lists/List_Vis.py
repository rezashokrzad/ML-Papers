import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go

#read dataframes of adjectives
df1 = pd.read_excel('Hotel_df_list1.xlsx')
df2 = pd.read_excel('Hotel_df_list2.xlsx')
df3 = pd.read_excel('Hotel_df_list3.xlsx')

#receive tooltip content
tooltip_columns = ['text', 'idx', 'Label']
tooltip_list = df1[tooltip_columns].values.tolist()

#dash setup
app = dash.Dash()
dict_main = {'df_lsit1': df1, 'df_list2': df2, 'df_list3': df3}
data = list(dict_main.keys())


dict_method = {'PCA_BERT': 'PCA_BERT', 'UMP_BERT': 'UMP_BERT',
               'LDA_BERT': 'LDA_BERT', 'TSNE_BERT': 'TSNE_BERT'}
methods = list(dict_method.keys())


dict_label = {'Label': 'Label', 'simple_majority_voting': 'simple_majority_voting',
              'weighted_majority_voting': 'weighted_majority_voting'}
labels = list(dict_label.keys())

dict_comp_x = {'Dim1': 'x_1', 'Dim2': 'x_2', 'Dim3': 'x_3', 'Dim4': 'x_4', 'Dim5': 'x_5'}
comp_x = list(dict_comp_x.keys())

dict_comp_y = {'Dim2': 'y_2', 'Dim1': 'y_1', 'Dim3': 'y_3', 'Dim4': 'y_4', 'Dim5': 'y_5'}
comp_y = list(dict_comp_y.keys())

hovertext = "%{customdata[0]}<br>idx=%{customdata[1]}<br>Label=%{customdata[2]}<extra></extra>"
clr_scale = 'Rainbow'
scatter_plot = go.Scatter(x=df1["PCA_BERT_Dim1"],
                          y=df1["PCA_BERT_Dim2"],
                          customdata=tooltip_list,
                          mode='markers',
                          marker={'size':8, 'colorscale':clr_scale},
                          marker_color= df1['Label'],
                          hovertemplate=hovertext,
                          #name='clean',
                          showlegend=False)
dcc_graph = dcc.Graph(id='Main-Graph',
                      figure=go.Figure(data=[scatter_plot], layout={'legend':{'x':1, 'y':1},
                                                                    'showlegend': True}
                                       ).update_layout(
                                                        template="presentation", 
                                                        xaxis_title_text='Dim1',
                                                        yaxis_title_text='Dim2',
                                                        height=750,
                                                        legend={'title': 'Label'},
                                                        title='plot_title',
                                                        plot_bgcolor='lightgrey',
                                                        # paper_bgcolor='lightgrey',
                                                        margin=dict(t=100, b=80, l=300, r=300),
                                                        ))


app.layout = html.Div([
    html.H1('Hotel Embeddings Visualization', style={'text-align': 'center'}),
    html.Div([html.Label("Adjective-DF"), dcc.Dropdown(id='data-dropdown',
                               options=[{'label': label, 'value': label} for label in data],
                               value=list(dict_main.keys())[0],
                               multi=False,
                               searchable=False)], style={'width': '20%', 'display': 'inline-block'}),
        
    html.Div([html.Label("Method"), dcc.Dropdown(id='method-dropdown',
                               options=[{'label': label, 'value': label} for label in methods],
                               value=list(dict_method.keys())[0],
                               multi=False,
                               searchable=False)], style={'width': '20%', 'display': 'inline-block'}),
    html.Div([html.Label("Label-Sytem"), dcc.Dropdown(id='label-dropdown',
                               options=[{'label': label, 'value': label} for label in labels],
                               value=list(dict_label.keys())[0],
                               multi=False,
                               searchable=False)], style={'width': '20%', 'display': 'inline-block'}),
        
    html.Div([html.Label("X-Component"), dcc.Dropdown(id='x-axis-dropdown', 
                               options=[{'label': label, 'value': label} for label in comp_x],
                               value=list(dict_comp_x.keys())[0],
                               multi=False,
                               searchable=False)],style={'width': '20%', 'display': 'inline-block'}),
                                 
    html.Div([html.Label("Y-Component"), dcc.Dropdown(id='y-axis-dropdown', 
                               options=[{'label': label, 'value': label} for label in comp_y],
                               value=list(dict_comp_y.keys())[0],
                               multi=False,
                               searchable=False)],style={'width': '20%', 'display': 'inline-block'}),
        dcc_graph,
        ])

@app.callback(
    dash.dependencies.Output('Main-Graph', 'figure'),
    [dash.dependencies.Input('data-dropdown', 'value'),
     dash.dependencies.Input('method-dropdown', 'value'),
     dash.dependencies.Input('label-dropdown', 'value'),
     dash.dependencies.Input('x-axis-dropdown', 'value'),
     dash.dependencies.Input('y-axis-dropdown', 'value')],
    [dash.dependencies.State('Main-Graph', 'figure')])
def updateGraph(df_name, method_name, label_name, x_field, y_field, data):
    source = data['data']
    df = dict_main[df_name]
    source[0].update({'customdata': df[tooltip_columns].values.tolist()})
    with open("source.txt", "w") as text_file:
        text_file.write(str(source))
    with open("layout.txt", "w") as text_file:
        text_file.write(str(data['layout']))
        
    if x_field and y_field:
        source[0].update({'x': df[method_name+'_'+x_field].tolist(),
                          'y': df[method_name+'_'+y_field].tolist()})
        source[0].update({'marker':{'color': df[label_name].tolist(),
                                    'size':8, 'colorscale':clr_scale}})
        data['layout'].update({'title': {'text': str(method_name)+" | "+x_field+' vs '+y_field}})
        data['layout']['xaxis'] = {**data['layout']['xaxis'], 'title': {'text': x_field}}
        data['layout']['yaxis'] = {**data['layout']['yaxis'], 'title': {'text': y_field}}
    return {'data': source, 'layout': data['layout']}

if __name__ == '__main__':
    app.run_server(debug=True, host="0.0.0.0", port=1005)