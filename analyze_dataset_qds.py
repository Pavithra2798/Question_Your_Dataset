# pip install google-cloud-aiplatform  chromadb==0.3.26 pydantic==1.10.8 typing-inspect==0.8.0 typing_extensions==4.5.0 pandas datasets google-api-python-client transformers==4.33.1 pypdf faiss-cpu config #langchain==0.0.229
# pip install langchain 
### !pip install llama-index
# pip install pandas-profiling
# pip install openpyxl
# pip install seaborn
# pip install matplotlib
# pip install vertexai
# pip install pyspark
# sudo apt-get update
# sudo apt-get install default-jdk
# pip install typing-extensions --upgrade


# pip install regex
# pip install flask


# add it for cloud shell environment
# pip install google-cloud-aiplatform 

# run below command in Cloud shell terminal window if any quota error:

# gcloud auth application-default login
# gcloud auth application-default set-quota-project <project-id>

 
import re
import pandas as pd
from pyspark.sql.types import StructType
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, TimestampType
from pyspark.sql import SparkSession
# from ydata_profiling import ProfileReport
import matplotlib.pyplot as plt
import seaborn as sns
# import ipywidgets as widgets
# from ipywidgets import widgets, Layout
# from IPython.display import display

import os
from pathlib import Path
import shutil
from google.cloud import storage
# upload to bucket
# import PyPDF2
import os
import pathlib as pt 

import urllib
import warnings
from pathlib import Path as p
from pprint import pprint

import pandas as pd
from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import VertexAIEmbeddings
from langchain.llms import VertexAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
import vertexai
from langchain.chains import LLMChain

  
PROJECT_ID = "<project-id>"
REGION = "us-central1"

vertexai.init(project=PROJECT_ID, location=REGION)

warnings.filterwarnings("ignore")


vertex_llm_text = VertexAI(model_name="text-bison@001",max_output_tokens=1024)

df = pd.DataFrame()
  
def get_excel(uploaded_file, user_query):
    merged_df = None
    file_path = uploaded_file
    all_sheets = pd.read_excel(file_path, sheet_name=None, engine="openpyxl")
    
    num_sheets = len(all_sheets)
    print("Number of sheets in the Excel file:", num_sheets)

    # Check if data is available in more than one sheet
    if num_sheets == 1:
        print("Data is available in one sheet")
        merged_df = None
        for key, df in all_sheets.items():
            merged_df = df
    else:
        print("Data is available in more than one sheet")

        # # Check if all sheets contain data
        all_sheets_have_data = all(all_sheets[sheet_name].empty for sheet_name in all_sheets)
        print(all_sheets_have_data)
        # if ~all_sheets_have_data:
        #     print("Not all sheets contain data")
        #     merged_df = None
        #     for key, df in all_sheets.items():
        #         merged_df = df
        # else:
        print("all sheets contain data")
        for sheet_name, sheet_df in all_sheets.items():
            pass
            
        # Find the primary column
        common_columns = set(all_sheets[sheet_name].columns)
        for key, df in all_sheets.items():
            common_columns = common_columns.intersection(df.columns)

        if len(common_columns) == 0:
            print("No common primary column found")
        else:
            primary_column = common_columns.pop()
            print("Primary column found:", primary_column)

            # Merge the dataframes based on the primary column
            for key, df in all_sheets.items():
                if merged_df is None:
                    merged_df = df
                else:
                    merged_df = pd.merge(merged_df, df, on=primary_column, how='outer')
                        
    merged_df_300 = merged_df.head(1000) # taking only top 300 rows due to input token constraint
    merged_df_300 = merged_df_300.apply(lambda row: ','.join(map(str, row)), axis=1)
  
    spark = SparkSession.builder.appName("Questioning_Dataset").getOrCreate()
    schema = StructType([StructField(field, StringType(), True) if merged_df[field].dtype == 'O' else
                        StructField(field, IntegerType(), True) if merged_df[field].dtype == 'int64' else
                        StructField(field, TimestampType(), True) if merged_df[field].dtype == 'datetime64[ns]' else
                        StructField(field, FloatType(), True) for field in merged_df.columns])
    spark_df = spark.createDataFrame(merged_df, schema=schema)
    spark_df.createOrReplaceTempView("Data_Table")
    data_schema = spark_df.schema
    string_schema = ""
    for field in data_schema:
        string_schema = string_schema + " column_name "+ field.name + " data_type "+ str(field.dataType)

    sql_query, result_data, pattern, predictions = try_cache_sql_query(user_query, string_schema, spark, merged_df_300)
    spark.stop()
    return sql_query, result_data, pattern, predictions

def generate_sql(user_query,df_cols,ftype, merged_df_300):
    llm = VertexAI(
        model_name="text-bison-32k",
        max_output_tokens=8000,
        temperature=0.2,
        top_p=0.8,
        top_k=40,
        verbose=True,
    )
    table_columns = df_cols

    if ftype == "query": 
        prompt_template = PromptTemplate(
        input_variables=['table_columns','user_query'],
        template=f'''
            You are an AI designed to comprehend natural language questions and translate them into 
            SQL queries. You have expertise in utilizing Common Table Expressions (CTEs), 
            window functions, and ranking in SQL to handle intricate data retrieval tasks.
            Your task is to analyze user queries about a database and produce the 
            corresponding SQL statements that can be executed to retrieve the requested data. 
            The database schema is as follows:

            Data Schema: 
            {table_columns}
            Please read the user's question and create an SQL query that accurately captures the 
            essence of their request. When presented with a user's English question, create a SQL query 
            that leverages the necessary advanced SQL techniques to extract the requested information. 
            Make sure to use CTEs, window functions, quartile, ntile, lead, lag, ranking etc when the 
            retrieval process requires 
            grouping, partitioning, or ordering data etc in a complex manner.After generating the query, 
            review it to ensure it matches the 
            question's intent. If you think the query might not be fully accurate or if it can be improved,
            please provide an alternative or refined query. Your primary goal is to deliver the most 
            accurate and efficient SQL query in response to the user's question.

            Write the sql query with the table name 'Data_Table'.
            The SQL query should be supported in pyspark. The sql query should be in a single line without
            newlines and quotes. The sql query should start with ##ss and end with $$ss.
            Translate the following user's English question into an advanced SQL 
            query using the appropriate techniques:
            
            User's English Question: {user_query}

            Never forget to start the query with ##ss and end with $$ss
            '''
            )

    query = LLMChain(llm=llm, prompt=prompt_template, verbose=False)
    print(query)
    return query.run(user_query=user_query)
  
def data_visualization():
    if df.shape[1] <= 1:
        return
    column_dropdown1 = widgets.Dropdown(
        options=list(df.columns),
        description='Select Column 1:',
        disabled=False,
        style={'description_width': 'initial', 'width': '200px'}
    )

    column_dropdown2 = widgets.Dropdown(
        options=list(df.columns),
        description='Select Column 2:',
        disabled=False,
        style={'description_width': 'initial', 'width': '200px'}
    )
    chart_dropdown = widgets.Dropdown(
        options=['Bar Plot', 'Line Plot', 'Scatter Plot', 'Histogram', 'Box Plot'],
        description='Select Chart Type:',
        disabled=False,
        style={'description_width': 'initial', 'width': '200px'}
    )

    output = widgets.Output()

    def create_visualization(column1, column2, chart_type):
        with output:
            output.clear_output()
            plt.figure(figsize=(8, 6))
            if chart_type == 'Bar Plot':
                sns.barplot(x=column1, y=column2, data=df)
                plt.title(f'Bar Plot for {column1} vs {column2}')
            elif chart_type == 'Line Plot':
                sns.lineplot(x=column1, y=column2, data=df, marker='o')
                plt.title(f'Line Plot for {column1} vs {column2}')
            elif chart_type == 'Scatter Plot':
                sns.scatterplot(x=column1, y=column2, data=df)
                plt.title(f'Scatter Plot for {column1} vs {column2}')
            elif chart_type == 'Histogram':
                sns.histplot(df[column1], kde=True)
                plt.title(f'Histogram of {column1}')
            elif chart_type == 'Box Plot':
                sns.boxplot(x=column1, y=column2, data=df)
                plt.title(f'Box Plot of {column1} vs {column2}')
            plt.show()

    def on_submit_button_clicked(b):
        create_visualization(column_dropdown1.value, column_dropdown2.value, chart_dropdown.value)

    display(column_dropdown1, column_dropdown2, chart_dropdown, output)

    column_dropdown1.observe(on_submit_button_clicked, names='value')
    column_dropdown2.observe(on_submit_button_clicked, names='value')
    chart_dropdown.observe(on_submit_button_clicked, names='value')
 

def try_cache_sql_query(user_query, string_schema, spark,merged_df_300,ftype="query"):
    n = 0
    while True or n <= 10:
        try:
            llm_result = generate_sql(user_query, string_schema,ftype, merged_df_300)

            llm_patterns = generate_sql(user_query, string_schema,"patterns", merged_df_300)

            if "I cannot understand the user question" in llm_result:
                print("This question doesn't match the context")
                break
            result = re.search(r'##ss(.*?)\$\$ss', llm_result, re.DOTALL)
            print("&&&&&&",llm_result)
            if result:
                sql_query = result.group(1)
                print("regex")               
            else:
                print("regex error")
                x = 1/0
            
            result2 = spark.sql(sql_query)
            print("SQLLLL", sql_query)  
            result_list = llm_patterns.split("*****")
            # print("@@@@@",result_list[0], result_list[1])
            global df
            df = result2.toPandas()
            return sql_query, df, result_list[0], result_list[1]
            break  # If the code runs without error, exit the loop
        except Exception as e:
            print("An error occurred:", e)
            n+=1

 



