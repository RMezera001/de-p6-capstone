# Do all imports and installs here
from pyspark.sql import SparkSession
import numpy as np
import os
import logging
from pyspark.sql.functions import split
import pyspark.sql
from pyspark.sql.functions import isnan, when, count, col, regexp_replace, lower, trim, to_date
from pyspark.sql.types import DoubleType,IntegerType
import pyspark.sql.functions as F

spark = SparkSession.builder.\
config("spark.jars.packages","saurfang:spark-sas7bdat:2.0.0-s_2.11")\
.enableHiveSupport().getOrCreate()


############# Airport data
bucket = 's3://de-capstone/raw_data/'

######### Airport code data
# Read in the data here
data = bucket + 'airport-codes_csv.csv'
df_apt_codes = spark.read.csv(data,header=True)

# pull just the airports in the US
df_apt_codes = df_apt_codes.filter(df_apt_codes.iso_country=='US')

# remove iata code == null
df_apt_codes = df_apt_codes.filter(df_apt_codes.iata_code.isNotNull())

# Get state code from iso_region
df_apt_codes = df_apt_codes.withColumn('State_Code',split(df_apt_codes.iso_region,'-').getItem(1))

######### temp data
data = bucket + 'temp_data.csv'
df_temp = spark.read.csv(data,header=True)

# Change AverageTemperature to Double
df_temp = df_temp.withColumn("AverageTemperature", df_temp["AverageTemperature"].cast("double"))
# filter just United states
df_temp = df_temp.filter(df_temp.Country == 'United States')
# drop nulls
df_temp = df_temp.where(col('AverageTemperature').isNotNull())

# get average temp for each city
avg_temp = df_temp.cube(['City','Latitude','Longitude']).agg({'AverageTemperature':'avg'})
avg_temp = avg_temp.na.drop()
avg_temp = avg_temp.withColumnRenamed('avg(AverageTemperature)', 'AverageTemperature')

########## Demographic data
data = bucket + 'us-cities-demographics.csv'
df_demo = spark.read.csv(data, header=True,sep=';')

# Combine temp data
df_demo = df_demo.join(avg_temp,avg_temp.City==df_demo.City).select(df_demo["*"],avg_temp["AverageTemperature"])

# Break data down to state level
state_df = df_demo.cube('State','State Code').agg({'Male Population':'sum',
                                     'Female Population':'sum',
                                     'Total Population':'sum',
                                     'Number of Veterans':'sum',
                                     'Foreign-born':'sum',
                                     'Average Household Size':'avg',
                                     'AverageTemperature':'avg'})
state_df = state_df.na.drop()
state_df = state_df.withColumnRenamed('avg(AverageTemperature)', 'AverageTemperature')
state_df = state_df.withColumnRenamed('avg(Average Household Size)', 'Average_Household_Size')
state_df = state_df.withColumnRenamed('sum(Total Population)', 'Total_Population')
state_df = state_df.withColumnRenamed('sum(Female Population)', 'Female_Population')
state_df = state_df.withColumnRenamed('sum(Male Population)', 'Male_Population')
state_df = state_df.withColumnRenamed('sum(Number of Veterans)', 'Number_of_Veterans')
state_df = state_df.withColumnRenamed('sum(Foreign-born)', 'Foreign-born')

# Sum up Races for each state
white_df = df_demo.where(df_demo.Race =='White').cube('State').agg({'Count':'sum'})
white_df = white_df.withColumnRenamed('sum(Count)', 'White')

asian_df = df_demo.where(df_demo.Race =='Asian').cube('State').agg({'Count':'sum'})
asian_df = asian_df.withColumnRenamed('sum(Count)', 'Asian')

black_df = df_demo.where(df_demo.Race =='Black or African-American').cube('State').agg({'Count':'sum'})
black_df = black_df.withColumnRenamed('sum(Count)', 'Black_or_African-American')

hispanic_df = df_demo.where(df_demo.Race =='Hispanic or Latino').cube('State').agg({'Count':'sum'})
hispanic_df = hispanic_df.withColumnRenamed('sum(Count)', 'Hispanic_or_Latino')

native_df = df_demo.where(df_demo.Race =='American Indian and Alaska Native').cube('State').agg({'Count':'sum'})
native_df = native_df.withColumnRenamed('sum(Count)', 'American_Indian_and_Alaska_Native')

# Combine race data
state_df = state_df.join(white_df,white_df.State==state_df.State).select(state_df["*"],white_df["White"])
state_df = state_df.join(asian_df,asian_df.State==state_df.State).select(state_df["*"],asian_df["Asian"])
state_df = state_df.join(black_df,black_df.State==state_df.State).select(state_df["*"],black_df["Black_or_African-American"])
state_df = state_df.join(hispanic_df,hispanic_df.State==state_df.State).select(state_df["*"],hispanic_df["Hispanic_or_Latino"])
state_df = state_df.join(native_df,native_df.State==state_df.State).select(state_df["*"],native_df["American_Indian_and_Alaska_Native"])

############# Sub tables
data = bucket + 'i94cit_res_codes.csv'
i94cit_res_codes = spark.read.csv(data, header=False)

i94cit_res_codes = i94cit_res_codes.withColumnRenamed('_c0', 'i94cit')
i94cit_res_codes = i94cit_res_codes.withColumnRenamed('_c1', 'i94_City')

#Copy df,rename columns, and join
res_codes = i94cit_res_codes
res_codes = res_codes.withColumnRenamed('i94cit','i94res')
res_codes = res_codes.withColumnRenamed('i94_City','i94_Resident')

#Join
i94cit_res_codes = i94cit_res_codes.join(res_codes,i94cit_res_codes.i94cit==res_codes.i94res)
#change data type
i94cit_res_codes = i94cit_res_codes.withColumn("i94cit", i94cit_res_codes["i94cit"].cast("double"))
i94cit_res_codes = i94cit_res_codes.withColumn("i94res", i94cit_res_codes["i94res"].cast("double"))

data = bucket + 'i94port_code.csv'
i94port_df = spark.read.csv(data, header=False)

# Drop empty column
i94port_df = i94port_df.drop('_c2')

# Remove spaces
i94port_df = i94port_df.select('_c0',trim(col('_c1')))

# Change column names
i94port_df = i94port_df.withColumnRenamed('_c0','i94port')
i94port_df = i94port_df.withColumnRenamed('trim(_c1)','location')

# get port city and state
i94port_df = i94port_df.withColumn('Port_City',split(i94port_df.location,',').getItem(0))
i94port_df = i94port_df.withColumn('Port_State',split(i94port_df.location,',').getItem(1))

data = bucket + 'i94addr.csv'
i94addr = spark.read.csv(data, header=False)

# Change column names
i94addr = i94addr.withColumnRenamed('_c0','State Code')
i94addr = i94addr.withColumnRenamed('_c1','State')

data = bucket + 'visa_data.csv'
df_visa = spark.read.csv(data, header=True)

# Drop empty column
df_visa = df_visa.drop('_c0')
# change data type
df_visa = df_visa.withColumn("i94visa", df_visa["i94visa"].cast("double"))

data = bucket + 'mode_data.csv'
df_mode = spark.read.csv(data, header=True)

# Drop empty column
df_mode = df_mode.drop('_c0')
# change data type
df_mode = df_mode.withColumn("i94mode", df_mode["i94mode"].cast("double"))

############## Immigration data
data = bucket + 'sas_data/'
df_imm =spark.read.format('com.github.saurfang.sas.spark').parquet(data)

# Change data types to Double
double_type = [
    'cicid','i94yr','i94mon','admnum'
]
for c in double_type:
    df_imm = df_imm.withColumn(c, df_imm[c].cast(DoubleType()))

# change data types to integer
int_type = [
    'i94yr','i94mon','i94bir','biryear'
]
for c in int_type:
    df_imm = df_imm.withColumn(c, df_imm[c].cast(IntegerType()))

#change data to date
date_type = [
    'dtadfile','dtaddto'
]
for c in date_type:
    df_imm = df_imm.withColumn(c, to_date(df_imm[c],'yyyyMMdd'))

############### Combining data
# merge immigration and df_visa data
staging_df = df_imm.join(df_visa,df_visa.i94visa==df_imm.i94visa,how='left').select(df_imm["*"],df_visa["Visa"])

# merge immigration and df_mode data
staging_df = staging_df.join(df_mode,df_mode.i94mode==staging_df.i94mode,how='left').select(staging_df["*"],df_mode["Mode"])

# merge immigration and i94port_df data
staging_df = staging_df.join(i94port_df,i94port_df.i94port==staging_df.i94port,how='left').select(staging_df["*"],i94port_df["Port_City"])
staging_df = staging_df.join(i94port_df,i94port_df.i94port==staging_df.i94port,how='left').select(staging_df["*"],i94port_df["Port_State"])


# merge immigration and i94cit_res_codes data
staging_df = staging_df.join(i94cit_res_codes,i94cit_res_codes.i94cit==staging_df.i94cit,how='left').select(staging_df["*"],i94cit_res_codes["i94_City"])
staging_df = staging_df.join(i94cit_res_codes,i94cit_res_codes.i94res==staging_df.i94res,how='left').select(staging_df["*"],i94cit_res_codes["i94_Resident"])

# merge immigration and i94addr data
staging_df = staging_df.join(i94addr,i94addr["State Code"]==staging_df["i94addr"],how='left').select(staging_df["*"],i94addr["State"])


# Combine staging, demo, and airport code data
final_df = staging_df.join(state_df,lower(state_df["State"])==lower(staging_df["State"]),how='left').drop(state_df["State Code"]).drop(state_df["State"])
final_df = final_df.join(df_apt_codes,df_apt_codes["iata_code"]==final_df["i94port"],how='left')

################ Quality checks
# Perform quality checks here
print('Performing Qulity Checks')
assert state_df.where(col('State').isNull()).count()==0,'Null values in state_df.State'
assert df_imm.where(col('admnum').isNull()).count()==0,'Null values in df_imm.admnum'
assert df_apt_codes.where(col('ident').isNull()).count()==0,'Null values in df_apt_codes.ident'
assert final_df.count() > 1000000,'final_df has less than 1 million records'
assert df_imm.count() > 1000000,'df_imm has less than 1 million records'
print('Quality checks passed')

########### Write data
print('Writing data to s3')
# Write data lake
output = 's3://de-capstone/datalake/'
final_df.write.mode('overwrite').partitionBy('State').parquet(output+'immigration_data')

# write file for data warehouse
output = 's3://de-capstone/datawarehouse/'
state_df.write.mode("overwrite").csv(output+'state_data')
staging_df.write.mode("overwrite").partitionBy('State').csv(output+'immigration_data')
df_apt_codes.write.mode("overwrite").csv(output+'airport_data')

print('de-capstone-etl.py finished!')
