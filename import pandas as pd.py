import pandas as pd
dfs = []  # List to store the DataFrames

for i in range(1, 9):
   file_name = f'RAT0{i}.csv'  # Construct the file name
   df = pd.read_csv(file_name)  # Read the CSV file into a DataFrame
   df['Target'] = file_name[:-4]# Add an 'Target' column with values as filename without extection
   if i == 1 :
      if  '147.32.83.234' in df['Source']  or  '147.32.83.234' in df['Destination'] :
         df['Malious'] = 1
      else:
         df['Malious']=0
   if i == 2:
      if  '147.32.83.253' in df['Source'] or  '147.32.83.253' in df['Destination'] :
         df['Malious'] = 1
      else:
         df['Malious']=0
   if i == 3:
      if  '35.201.97.85' in df['Source'] or  '35.201.97.85' in df['Destination'] :
         df['Malious'] = 1
      else:
         df['Malious']=0
   if i == 4:
      if  '147.32.83.181' in df['Source'] or  '147.32.83.181' in df['Destination'] :
         df['Malious'] = 1
      else:
         df['Malious']=0
   if i == 5 :
      if  '147.32.83.234 ' in df['Source'] or  '147.32.83.234' in df['Destination'] :
         df['Malious'] = 1
      else:
         df['Malious']=0
   if i == 6 :
      if  '192.168.131.1' in df['Source'] or  '192.168.131.1' in df['Destination'] :
         df['Malious'] = 1
      else: 
         df['Malious']=0
   if i == 7 :
      if  '147.32.83.230' in df['Source'] or  '147.32.83.230' in df['Destination']:
         df['Malious'] = 1
      else:
         df['Malious']=0
   if i == 8 :
      if  '147.32.83.157' in df['Source'] or  '147.32.83.157' in df['Destination']:
         df['Malious'] = 1 
      else:
         df['Malious']=0   
   dfs.append(df)  # Append the DataFrame to the list
df_merge = pd.concat(dfs,ignore_index=True)
# Print the first few rows of the DataFrame where 'Malicious' is equal to 1
df_merge.head(200000)

# dfs =[df,df1,df2,df3,df4,df5,df6,df7]
# df_merge = pd.concat(dfs,ignore_index=True)   