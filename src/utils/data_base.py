class DataInfo:
    @classmethod
    def info(cls, df): 
        
        # Counting no of rows 
        print(f'\nTotal Rows : {df.shape[0]} \n' + '--'*10 )
      
        # Counting no of columns
        print(f'\nTotal Columns : {df.shape[1]} \n' + '--'*10)
        
        # Extracting column names
        column_name =  df.columns 
        print(f'\nColumn Names\n' + '--'*10 +  f'\n{column_name} \n \n')
        
        # Data type info
        print(f'Data Summary\n' + '--'*10)
        data_summary = df.info() 
        
        # Total null values by each categories
        null_values = df.isnull().sum() 
        print(f'\nNull values\n' + '--'*10 + f'\n{null_values} \n \n')

        # Descriptive statistics
        describe =  df.describe() 
        print(f'\nDescriptive Statistics\n' + '--'*10 +  f'\n{describe} \n \n')
    
    @classmethod
    def analyze_unique_values(cls, df):      
        
        # Number of Unique values in each column
        uniques_values = df.apply(lambda x : len(x.unique())) 
        print(f'Unique Values\n' + '--'*10 +  f'\n{uniques_values} \n \n')
        
        for i in df.columns: 
          
          # Columns having datatype object or categories in each column
            if df[i].dtype == 'O':          
              
              # Unique categories in each columns
                unique_category = set(df.loc[: , i])      
                
                if len(unique_category)>=100:
                  
                    #Taking only top 50 else it will will messed up the output
                    print(f'{i} -- {list(unique_category)[:50]}\n')
                    continue  
                else:
                    print(f'\n{i} -- {unique_category}\n')

class DataCleaner:

    @classmethod
    def clean_data(cls, data):
        # Basic cleaning applicable to all data types
        lowercase = lambda x: str(x).lower()
        data.rename(lowercase, axis='columns', inplace=True)
        return data