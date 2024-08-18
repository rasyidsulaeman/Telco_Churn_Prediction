import pandas as pd 
import plotly.graph_objects as go
import plotly.express as px

class TelcoEDA():

    def __init__(self, path):
        self.path = path

    def data_loaded(self):
        df = pd.read_csv(self.path)
        return df
    
    def missing_value(self, df):
        
        missing = []
        for columns in df.columns:
            missing_shape = df[df[columns].isna()].shape
            percentage = round(missing_shape[0]/df.shape[0] * 100, 3)

            if df[columns].dtypes != 'object':
                max, min = df[columns].max(), df[columns].min()
            else:
                max, min = '-', '-'
                
            missing.append({'Columns' : columns,
                            'N/A count' : missing_shape[0],
                            'Percentage' : str(percentage) + '%',
                            'Max' : max,
                            'Min' : min}
                            )
        return pd.DataFrame(missing).sort_values(by='Percentage', ascending=False)
    
    def total_missing(self):
        df = self.data_loaded()
        missing = self.missing_value(df)
        return round(missing['N/A count'].sum() / df.shape[0] * 100, 2)

    def total_duplicated(self):
        df = self.data_loaded()
        duplicated_percentage = round(df.duplicated().sum() / df.shape[0] * 100,2)
        return [df.duplicated().sum(), duplicated_percentage]
                                                                                          
    def remove(self, df, bool=False):
        if bool:
            # Drop missing value and duplicated value            
            df.dropna(inplace=True)
            df.drop_duplicates(inplace=True)

        return df



class TelcoPlot():

    def __init__(self, df=None):
        self.df = df

    def bivariate_category(self, choose):

        churn = self.df[self.df['Churn'] == 'Yes'][choose].value_counts()
        no_churn = self.df[self.df['Churn'] == 'No'][choose].value_counts()

        fig = go.Figure()
        fig.add_trace(go.Bar(x=churn.index, y=churn.values, 
                             name='Churn', text=churn.values))
        
        fig.add_trace(go.Bar(x=no_churn.index, y=no_churn.values, 
                             name='No Churn', text=no_churn.values))
        
        fig.update_layout(title_text=f'Number of {choose} according to Churn Classification',
                          xaxis_tickfont_size=14,
                          yaxis=dict(title='Count',titlefont_size=16,tickfont_size=14), 
                          barmode='stack', 
                          uniformtext_minsize=12, uniformtext_mode='hide')
        
        return fig
    
    def bivariate_numeric(self, choose):

        fig = px.histogram(self.df, x=choose, color="Churn", 
                           marginal="box", # or violin, rug
                           hover_data=self.df.columns, text_auto=True)
        
        fig.update_layout(title_text=f'Number of {choose} according to Churn Classification',
                          xaxis_tickfont_size=14,
                          yaxis=dict(title='Count',titlefont_size=16,tickfont_size=14), 
                          barmode='stack', 
                          uniformtext_minsize=12, uniformtext_mode='hide')
        
        return fig
    


    
    


