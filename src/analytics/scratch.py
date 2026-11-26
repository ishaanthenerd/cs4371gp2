from pathlib import Path
 
PANDAS_FILEPATH = 'T1_seattle_incidents.csv'
directory = Path('.')
for file in directory.iterdir():
    if file.is_file():
        print(file.name)
        print(file.name == PANDAS_FILEPATH)