## Democratizing Data Analysis with Natural Language Queries
 

Data analysis is often a hurdle for non-technical users, hindering their ability to extract valuable insights from complex datasets. To bridge this gap, a system was developed to enable natural language querying of data. The task involved creating a user-friendly tool that allows users to ask questions and receive real-time answers, empowering them to explore data and uncover hidden patterns.

 

Leveraging Google Cloud's Vertex AI, "text-bison-32k" model and LangChain, the system allows users to upload Excel datasets and query them in plain language. A carefully crafted prompt instructs the LLM to translate user queries into optimized SQL code, employing advanced techniques like CTEs and window functions if needed. PySpark is then used to execute the generated SQL against the uploaded data, retrieving and presenting results. The LLM also analyzes the data to identify patterns and trends, displayed in a user-friendly format. The system is deployed as a Flask web application.

 

The prototype successfully demonstrates the ability to analyze datasets and answer complex natural language queries. It accurately retrieves data and identifies patterns, surpassing the capabilities of standard tools like Excel for complex analysis. This technology empowers non-technical users to independently explore their data, gain valuable insights, and make data-driven decisions.

## **Skills and Technology:**  
Python, PaLM 2, Prompt Engineering, Pandas, PySpark, Langchain, Cloud Build, Cloud Function, Google Cloud Storage, Vertex AI
