

# The code for RegGC

This repo contains the source code for the RegGC model.


## Programming language

Tensorflow

## Required libraries

numpy sklearn matplotlib pandas networkx sqlalchemy geopy==1.21.0

## Hardware info

GeForce GTX 1080 Ti 11 GB GPU

## Dataset info

* The first dataset (HK) is a taxi GPS data set in Hong Kong in 2010. It contains 35 gigabytes trajectories, and the updating rate of GPS points is around every 40 seconds. 
* The second dataset (XN) is an open GPS data set [1], which contains 137 gigabytes of GPS data in Xi’an city’s second ring road region in October and November of 2016 in China.
* The third dataset (CD) is an open GPS data set [1], which contains 196 gigabytes of GPS data in Chengdu second ring road region in October and November of 2016 in China.

HK dataset is confidential, which is provided by Prof. S.C. Wong of Civil Engineering in HKU; XN and CD datasets are open datasets from the didi company. You can go to the didichuxing website to submit a data access application for them.

## Data preprocessing

* After obtaining datasets, you can use preprocess_data.py for preprocessing. The original code of preprocess_data.py is provided by [2]. Thanks again for their sharing. We adapt their preprocessing code by adding temporal traffic data and context data.

* We use [3] to do the map matching. You could follow the instructions in the free tool: https://github.com/bmwcarit/barefoot. After finishing the map matching, you can calculate the travel speed for each edge. For SQL of Xi'an and Chengdu datasets, you could use python code to write your preprocessed records as tables in SQL:

    import pandas as pd

    from sqlalchemy import create_engine

    tablename = 'xian' or tablename = 'chengdu' 

    engine = create_engine("mysql+pymysql://root:dbname@localhost:3306/username", echo=True)

    df_sql = pd.DataFrame(gps_df_i, columns=["seg_id", "start_time", "travel_speed", "vehid", "len"])

    df_sql.to_sql(tablename, con=engine, if_exists='append')



## How to run the code for RegGC
```
python reggc_main.py
```

## References
* [1] http://outreach.didichuxing.com/research/opendata/
* [2] Jilin Hu, Chenjuan Guo, Bin Yang, Christian S. Jensen, Stochastic Weight Completion for Road Networks Using Graph Convolutional Networks (ICDE 2019)
* [3] P. Newson and J. Krumm, “Hidden markov map matching through noise and sparseness,” in SIGSPATIAL, 2009.
# RegGC
