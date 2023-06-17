import pandas as pd

def get_database(path, columns):
    df = pd.read_csv(path, usecols = columns, sep = ',', decimal = '.', low_memory = True)
    return df

###     CIGNO           ###
# data - ora - direzione - declinazione
path_database = "../Data/Cigno/Cigno.csv"
columns_list = ["Data", "Orario Milano", "Declinazione", "Direzione"]
df = get_database(path_database, columns_list)
tab = ""
sp = " & "
for index, row in df.iterrows():
    tab += row['Data'] + sp + row['Orario Milano'] + sp + row['Direzione'] + sp + row['Declinazione'] + " \\\\\n"

#print(tab)

###     ANDROMEDA       ###
path_database = "../Data/Andromeda/Andromeda.csv"
columns_list = ["Data", "Orario Milano"]
df = get_database(path_database, columns_list)
tab = ""
sp = " & "
for index, row in df.iterrows():
    tab += row['Data'] + sp + row['Orario Milano'] + " \\\\\n"

#print(tab)

###     CASSIOPEA       ###
path_database = "../Data/Cassiopea/Cassiopea.csv"
columns_list = ["Data", "Orario Milano"]
df = get_database(path_database, columns_list)
tab = ""
sp = " & "
for index, row in df.iterrows():
    tab += row['Data'] + sp + row['Orario Milano'] + " \\\\\n"

#print(tab)