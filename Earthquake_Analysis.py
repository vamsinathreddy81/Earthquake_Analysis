#!/usr/bin/env python
# coding: utf-8

# In[2]:


## Import the relevant libraries into the environment

import numpy as np                ## linear algebra
import pandas as pd               ## data processing, dataset file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt   ## data visualization & graphical plotting
import seaborn as sns             ## to visualize random distributions
import plotly.express as px       ## data visualization & graphical plotting
import squarify                   ## Treemap plots

get_ipython().run_line_magic('matplotlib', 'inline')
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode

pd.options.display.float_format = '{:.2f}'.format  ## limiting the decimals in the output to 2 

import warnings                    ## Filter warnings
warnings.filterwarnings('ignore')


# In[3]:


## Load the dataset 

df = pd.read_csv('earthquake_data.csv')

## Check a few records in the dataset that we just loaded

df.head(5)


# In[4]:


## Check the dataset shape, rows, columns, duplicate entries & missing values

print(f'\033[94mNumber of records (rows) in the dataset are: {df.shape[0]}')
print(f'\033[94mNumber of features (columns) in the dataset are: {df.shape[1]}')
print(f'\033[94mNumber of duplicate entries in the dataset are: {df.duplicated().sum()}')
print(f'\033[94mNumber missing values in the dataset are: {sum(df.isna().sum())}')


# In[5]:


## Let's find out which features have null values

df.isnull().sum()[df.isnull().sum() > 0]


# In[6]:


## Have a glance at the dataframe with info() and describe() functions

print('--'*40)
print(df.info())
print('--'*40, '\n', df.describe(include='all').T)
print('--'*40)


# In[7]:


## Let's visualise the Earthquakes by Magnitude, to understand it in a better way

sns.set(rc={'axes.facecolor':'none','axes.grid':False,'xtick.labelsize':13,'ytick.labelsize':13, 'figure.autolayout':True, 'figure.dpi':300, 'savefig.dpi':300})
my_col = ('#40E0D0', '#D2B48C','#c7e9b4', '#EEE8AA','#00FFFF','#FAEBD7','#FF6347', '#FAFAD2', '#D8BFD8','#F4A460','#F08080', '#EE82EE', '#4682B4','#6A5ACD', '#00C78C')

plt.subplots(figsize=(16,6))

plt.subplot(1,2,1)

plt.title('Earthquakes by Magnitude : Treemap',fontsize=16)
labels = df['magnitude'].value_counts().index.get_level_values(0).tolist()
sizes = df['magnitude'].value_counts().reset_index().magnitude.values.tolist()

squarify.plot(sizes=sizes, label=labels, color=my_col, alpha=0.3)
plt.axis('off')

plt.subplot(1,2,2)
plt.title('Earthquakes by Magnitude : Countplot', fontsize=16)
ax = sns.countplot(x='magnitude', data=df, palette=my_col, alpha=0.3)
for p in ax.patches: 
        ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+ p.get_width()/2, p.get_height() + 2), ha='center', va='bottom')
plt.ylabel(None), plt.yticks([]), plt.xlabel(None)
        
plt.show()


# In[8]:


## Let's understand the patterns in earthquake depth and significance
## We'll use histograms and box & whisker plots to identify the trend

plt.subplots(figsize=(14,10))

plt.subplot(2,2,1)
plt.title('Earthquakes by Depth : Histogram', pad=1, fontsize=16)
sns.distplot(df['depth'], color="#D2B48C", kde_kws={'linewidth':1,'color':'b'})
plt.yticks([]), plt.ylabel(None), plt.xlabel(None)

plt.subplot(2,2,2)
plt.title('Earthquakes by Significance : Histogram', pad=1, fontsize=16)
sns.distplot(df['sig'], color="#D2B48C", kde_kws={'linewidth':1,'color':'b'})
plt.yticks([]), plt.ylabel(None), plt.xlabel(None)

plt.subplot(2,2,3)
plt.title('Earthquakes by Depth : Box & Whisker Plot', pad=1, fontsize=16)
sns.boxplot(df['depth'], color="#c7e9b4", orient='v')
plt.ylabel(None), plt.xlabel(None), plt.xticks([])

plt.subplot(2,2,4)
plt.title('Earthquakes by Significance : Box & Whisker Plot', pad=1, fontsize=16)
sns.boxplot(df['sig'], color="#c7e9b4", orient='v')
plt.ylabel(None), plt.xlabel(None), plt.xticks([])

plt.show()


# In[9]:


## Let's understand the relationship between magnitude vs depth and magnitude vs Significance

plt.subplots(figsize=(14,6))
my_pal = ('#D2B48C','#40E0D0')
          
plt.subplot(1,2,1)
plt.title('Earthquakes by Magnitude  Vs Depth',fontsize=16)
sns.scatterplot(data=df, x='magnitude', y='depth', hue='tsunami', palette=my_pal)
plt.ylabel('Earthquake Depth', fontsize=15)
plt.xlabel('Earthquake Magnitude', fontsize=15)

plt.subplot(1,2,2)
plt.title('Earthquakes by Magnitude Vs Significance ',fontsize=16)
sns.scatterplot(data=df, x='magnitude', y='sig', hue='tsunami', palette=my_pal)
plt.ylabel('Earthquake Significance', fontsize=15)
plt.xlabel('Earthquake Magnitude', fontsize=15)

plt.show()


# In[10]:


## Let's visualise earthquakes by type of magnitude measurement used

plt.subplots(figsize=(14,6))

plt.subplot(1,2,1)
plt.title('Quakes by Type of Magnitude Measurement Used (Count)', fontsize=16)
ax = sns.countplot(y='magType', data=df, palette=my_col, alpha=0.3, order=df['magType'].value_counts().index)

for p in ax.patches:
    ax.annotate('{:.0f}'.format(p.get_width()),  (p.get_x() + p.get_width() + 10, p.get_y()+0.5))
plt.ylabel(None), plt.xlabel(None), plt.xticks([])

plt.subplot(1,2,2)
plt.title('Quakes by Type of Magnitude Measurement Used (% Count)',fontsize=16)
my_xpl = [0.0, 0.0, 0.0, 0.0, 0.1, 0.20, 0.30, 0.40, 0.50]
df['magType'].value_counts().plot(kind='pie', colors=my_col, explode=my_xpl, legend=None, ylabel='', counterclock=False, startangle=150, wedgeprops={'alpha':0.3, 'edgecolor' : 'white','linewidth': 0.5, 'antialiased': True}, autopct='%1.1f')

plt.show()


# In[11]:


## Make plots for Earthquakes by Data Contributor

plt.subplots(figsize=(14,6))

plt.subplot(1,2,1)
plt.title('Quakes by Data Contributor (net) (% Count)', fontsize=16)
my_xpl = [0.0, 0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
df['net'].value_counts().plot(kind='pie', colors=my_col, explode=my_xpl, legend=None, ylabel='', counterclock=False, startangle=150, wedgeprops={'alpha':0.3, 'edgecolor' : 'white','linewidth': 1, 'antialiased': True}, autopct='%1.1f')

plt.subplot(1,2,2)
plt.title('Quakes by Data Contributor (net) (Count)', fontsize=16)
ax = sns.countplot(x="net", data=df, palette=my_col, alpha=0.3, order=df['net'].value_counts().index)
for p in ax.patches: 
        ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+ p.get_width()/2, p.get_height() + 5), ha='center', va='bottom')
plt.yticks([]), plt.ylabel(None), plt.xlabel(None)

plt.show()


# In[12]:


## Understand Tsunami Occurence during earthquake events

plt.subplots(figsize=(12,4))

plt.subplot(1,2,1)
plt.title('Quakes by Tsunami Occurence (in %)',fontsize=14)
my_xpl = [0.0, 0.05]
df['tsunami'].value_counts().plot(kind='pie', colors=my_col, explode=my_xpl, legend=None, ylabel='', counterclock=False, startangle=150, wedgeprops={'alpha':0.2, 'edgecolor' : 'black','linewidth': 3, 'antialiased': True}, autopct='%1.1f')
plt.xlabel('Tsunami Occurence (0 = No, 1 = Yes)',fontsize=12)

plt.subplot(1,2,2)
plt.title('Quakes by Tsunami Occurence (Count)',fontsize=14)
ax = sns.countplot(y='tsunami', data=df, facecolor=(1,1,1,1), linewidth=4, edgecolor=sns.color_palette(my_col, 2), order=df['tsunami'].value_counts().index)

for p in ax.patches:
    ax.annotate('{:.0f}'.format(p.get_width()),  (p.get_x() + p.get_width() + 10, p.get_y()+0.5))

plt.xlabel('Tsunami Occurence (0 = No, 1 = Yes)',fontsize=12)
plt.xticks([]), plt.ylabel(None)
    
plt.show()


# In[13]:


## Earthquakes by Tsunami Occurence, by  Magnitude Measurement

plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.title('Tsunami Occurence, by Magnitude Measurement Type', pad=20, fontsize=14)
ax = sns.countplot(data=df, x='tsunami', hue='magType', palette=my_col)

for p in ax.patches:
   ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.02, p.get_height()+5))
plt.ylabel(None), plt.yticks([])
plt.xlabel('Tsunami Occurence (0 = No, 1 = Yes)',fontsize=12)

plt.subplot(1,2,2)
plt.title('Tsunami Occurence, by Magnitude Measurement Type & by Magnitude', pad=20, fontsize=14)
sns.boxplot(data=df, x="tsunami", y="magnitude", hue="magType", palette=my_col)
plt.xlabel('Tsunami Occurence (0 = No, 1 = Yes)',fontsize=12)

plt.show()


# In[14]:


## Earthquakes by Tsunami Occurence, by  Data Contributor (net)

plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.title('Tsunami Occurence, by  Data Contributor (net)', pad=20, fontsize=14)
ax = sns.countplot(x='tsunami', hue='net', palette=my_col, data=df)

for p in ax.patches:
   ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.02, p.get_height()+5))
plt.ylabel(None), plt.yticks([])
plt.xlabel('Tsunami Occurence (0 = No, 1 = Yes)',fontsize=12)

plt.subplot(1,2,2)
plt.title('Tsunami Occurence, by DataContributor & by Magnitude', pad=20, fontsize=14)
sns.boxplot(data=df, x="tsunami", y="magnitude", hue="net", palette=my_col)
plt.ylabel('Magnitude')
plt.xlabel('Tsunami Occurence (0 = No, 1 = Yes)', fontsize=12)

plt.show()


# In[15]:


## Analyse the data by Year, Month, Day of the month, and Hour of the day.

## Make a new Date, Year, Month, Day, & Hour columns from the 'date_time' column for our analysis

df['Date'] = pd.to_datetime(df['date_time']).dt.date
df['Year'] = pd.to_datetime(df['date_time']).dt.year
df['Month'] = pd.to_datetime(df['date_time']).dt.month
df['Day'] = pd.to_datetime(df['date_time']).dt.day
df['Hour'] = pd.to_datetime(df['date_time']).dt.hour

## Check the new columns we have created in our dataset

df[['date_time', 'Date', 'Year', 'Month', 'Day', 'Hour']].head(5)


# In[16]:


## Now, let's visualise the earthquake events by Year, Month, Day, and Hour with Scatterplots & Lineplots

plt.subplots(figsize=(12,14))

## By Year
###########

plt.subplot(421)
plt.title('Quakes by Year : Scatterplot', pad = 10, fontsize = 14)
sns.scatterplot(data = df['Year'].value_counts().sort_values(), color='g')
plt.ylim(0, 70), plt.ylabel(None), plt.xlabel(None)

plt.subplot(422)
plt.title('Quakes by Year : Lineplot', pad = 10, fontsize = 14)
sns.lineplot(data = df['Year'].value_counts().sort_values(), color='g', linewidth = 2)
plt.ylim(0, 70), plt.ylabel(None), plt.xlabel(None)

## By Month
###########

plt.subplot(423)
plt.title('Quakes by Month : Scatterplot', pad = 10, fontsize = 14)
sns.scatterplot(data = df['Month'].value_counts().sort_values(), color='g')
plt.ylim(0, 110), plt.ylabel(None), plt.xlabel(None)

plt.subplot(424)
plt.title('Quakes by Month : Lineplot', pad = 10, fontsize = 14)
sns.lineplot(data = df['Month'].value_counts().sort_values(), color='g', linewidth = 2)
plt.ylim(0, 110), plt.ylabel(None), plt.xlabel(None)

## By Day
###########

plt.subplot(425)
plt.title('Quakes by Day : Scatterplot', pad = 10, fontsize = 14)
sns.scatterplot(data = df['Day'].value_counts().sort_values(), color='g')
plt.ylim(0, 60), plt.ylabel(None), plt.xlabel(None)

plt.subplot(426)
plt.title('Quakes by Day : Lineplot', pad = 10, fontsize = 14)
sns.lineplot(data = df['Day'].value_counts().sort_values(), color='g', linewidth = 2)
plt.ylim(0, 60), plt.ylabel(None), plt.xlabel(None)

## By Hour
###########

plt.subplot(427)
plt.title('Quakes by Hour : Scatterplot', pad = 10, fontsize = 14)
sns.scatterplot(data = df['Hour'].value_counts().sort_values(), color='g')
plt.ylim(0, 60), plt.ylabel(None), plt.xlabel(None)

plt.subplot(428)
plt.title('Quakes by Hour : Lineplot', pad = 10, fontsize = 14)
sns.lineplot(data = df['Hour'].value_counts().sort_values(), color='g', linewidth = 2)
plt.ylim(0, 60), plt.ylabel(None), plt.xlabel(None)

plt.show()


# In[17]:


## Now we'll visualise the data by country
## But the country column has many missing values, we'll fix them and plot the data

## Let's fill the missing values in country column with information available in the location column

## Create a new dataframe using the location information in the earthquake dataset
## Make it into a two column dataframe by moving the last word of the location information in to the second (last) column
## Now the second column of new dataframe has country name, which we use to fill in our original dataset

df_r = df['location'].str.split(pat=',', n=1, expand=True)
print(df_r.head(5))                       ## View the new dataframe

## fill the missing country data, with country names in country column of df_r (df_r[1] is the country column)

df['country'] = df['country'].fillna(df_r[1])  
print('\n', 'Missing values in the refined country column are : ', df['country'].isna().sum())


# In[18]:


# Making a fresh dataframe by droping null values from 'country','location','continent' columns

df_country=df.dropna(subset=['country','location','continent'], how='all')
print('\n', 'Missing values in the country column are : ', df_country['country'].isnull().sum())


# In[19]:


## Now we check the missing values in location column, and label them as 'unknown'

print('\n', 'Missing location values Before : ', df_country['location'].isnull().sum())
df_country['location']=df_country['location'].fillna('unknown')
print('\n', 'Missing location values After  : ', df_country['location'].isnull().sum())


# In[20]:


## We try to fill location names as missing country names, and refine it later

df_country['country'] = df_country['country'].fillna(df_country['location'])

## Check the null values in country column

df_country['country'].isnull().sum()


# In[21]:


## Let's have a look at the country names in the country column

df_country['country'].unique()


# In[22]:


## Now we refine the column, for analysing and visualising the data country-wise

## First we remove all the extra spaces in the country column (in fact, the code will erase all the extra spaces across the dataset)

df_country = df_country.applymap(lambda x: x.strip() if isinstance(x, str) else x)

## Next we refine some country names

df_country.replace({'country': {"the Fiji Islands" : "Fiji region", "Fiji" : "Fiji region", 
                                "the Kermadec Islands" : "New Zealand region", "the Loyalty Islands" : "New Caledonia", 
                                "Vanuatu" : "Vanuatu region", "South Sandwich Islands" : "South Sandwich Islands region", 
                                "South Georgia and the South Sandwich Islands" : "South Sandwich Islands region", 
                                "Prince Edward Islands region" : "Canada", "Okhotsk" : "Russia region", 
                                "off the west coast of northern Sumatra" : "Indonesia", 
                                "Philippine Islands region" : "Philippines", 
                                "the Kuril Islands" : "Kuril Islands", 
                                "United Kingdom of Great Britain and Northern Ireland (the)": "UK", 
                                "People's Republic of China": "CHINA", "United States of America": "USA", 
                                "Alaska": "USA", "Aleutian Islands, Alaska" : "USA", "California" : "USA", 
                                "India" : "India region", "Russia" : "Russia region", "New Zealand" : "New Zealand region", 
                                "Japan region" : "Japan"}}, inplace=True)

df_country['country'].unique()


# In[23]:


## Now We plot country-wise data with barplots for all the countries and for top 20 countries

plt.subplots(figsize=(12,14))

plt.subplot(211)
plt.title('Quakes by Country - All (Count)', fontsize=16)
ax = sns.countplot(x=df_country['country'], palette='Greens_r', alpha=1, order=df_country['country'].value_counts().index)

for p in ax.patches: 
        ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.05, p.get_height()+3))

plt.xticks(rotation=90), plt.xlabel(None), plt.ylabel(None), plt.yticks([])

plt.subplot(212)
plt.title('Quakes by Country - Top 20 (Count)', fontsize=16)
ax = sns.countplot(x=df_country['country'], palette='Greens', alpha=1, order=df_country['country'].value_counts().head(20).index)

for p in ax.patches: 
        ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.05, p.get_height()+3))

plt.xticks(rotation=90), plt.xlabel(None), plt.ylabel(None), plt.yticks([])

plt.show()


# In[24]:


## From the above country-wise analysis we saw that Indonesia, Japan, and Papua New Guinea have highest number of quakes 
## Let's analyse these three most quake prone countries through appropriate visuals 

plt.subplots(figsize=(16,16))
plt.suptitle('Top 3 Earthquake Prone Countries in the World - Indonesia, Japan & Papua New Guinea', fontsize=23)

plt.subplot(3,3,1)
plt.title('Indonesia - Quakes by Tsunami Occurence (%)', pad=20, fontsize=16)
df_country[df_country['country'] == "Indonesia"]['tsunami'].value_counts().plot(kind='pie', subplots=True, colors = my_col, legend=None, ylabel='', wedgeprops={'alpha':1, 'edgecolor' : 'black','linewidth': 1, 'antialiased': True}, autopct='%1.1f%%')
plt.xlabel('Tsunami Occurence (0 = No, 1 = Yes)', fontsize=14)

plt.subplot(3,3,2)
plt.title('Japan - Quakes by Tsunami Occurence (%)', pad=20, fontsize=16)
df_country[df_country['country'] == "Japan"]['tsunami'].value_counts().plot(kind='pie', subplots=True, colors = my_col, legend=None, ylabel='', wedgeprops={'alpha':0.6, 'edgecolor' : 'black','linewidth': 1, 'antialiased': True}, autopct='%1.1f%%')
plt.xlabel('Tsunami Occurence (0 = No, 1 = Yes)', fontsize=14)

plt.subplot(3,3,3)
plt.title('Papua New Guinea - Quakes by Tsunami Occurence (%)', pad=20, fontsize=16)
df_country[df_country['country'] == "Papua New Guinea"]['tsunami'].value_counts().plot(kind='pie', subplots=True, colors = my_col, legend=None, ylabel='', wedgeprops={'alpha':0.3, 'edgecolor' : 'black','linewidth': 1, 'antialiased': True}, autopct='%1.1f%%')
plt.xlabel('Tsunami Occurence (0 = No, 1 = Yes)', fontsize=14)

plt.subplot(3,3,4)
plt.title('Quakes in Indonesia by Magnitude : Treemap', pad=20, fontsize=16)
labels = df_country[df_country['country'] == "Indonesia"]['magnitude'].value_counts().index.get_level_values(0).tolist()
sizes = df_country[df_country['country'] == "Indonesia"]['magnitude'].value_counts().reset_index().magnitude.values.tolist()

squarify.plot(sizes=sizes, label=labels, color=my_col, alpha=1)
plt.axis('off')

plt.subplot(3,3,5)
plt.title('Quakes in Japan by Magnitude : Treemap', pad=20, fontsize=16)
labels = df_country[df_country['country'] == "Japan"]['magnitude'].value_counts().index.get_level_values(0).tolist()
sizes = df_country[df_country['country'] == "Japan"]['magnitude'].value_counts().reset_index().magnitude.values.tolist()

squarify.plot(sizes=sizes, label=labels, color=my_col, alpha=0.6)
plt.axis('off')

plt.subplot(3,3,6)
plt.title('Quakes in Papua New Guinea by Magnitude : Treemap', pad=20, fontsize=16)
labels = df_country[df_country['country'] == "Papua New Guinea"]['magnitude'].value_counts().index.get_level_values(0).tolist()
sizes = df_country[df_country['country'] == "Papua New Guinea"]['magnitude'].value_counts().reset_index().magnitude.values.tolist()

squarify.plot(sizes=sizes, label=labels, color=my_col, alpha=0.3)
plt.axis('off')

plt.subplot(3,3,7)
plt.title('Quakes in Indonesia by Year: Barplot', pad=20, fontsize=16)
ax = sns.countplot(x='Year', data=df_country[df_country['country'] == "Indonesia"], color='#D2B48C', alpha=1)

for p in ax.patches: 
        ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+ p.get_width()/2, p.get_height() + 0.1), ha='center', va='bottom')
plt.ylabel(None), plt.yticks([]), plt.xticks(fontsize=10, rotation=90), plt.xlabel(None)

plt.subplot(3,3,8)
plt.title('Quakes in Japan by Year: Barplot', pad=20, fontsize=16)
ax = sns.countplot(x='Year', data=df_country[df_country['country'] == "Japan"], color='#D2B48C', alpha=0.6)

for p in ax.patches: 
        ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+ p.get_width()/2, p.get_height() + 0.1), ha='center', va='bottom')
plt.ylabel(None), plt.yticks([]), plt.xticks(fontsize=10, rotation=90), plt.xlabel(None)

plt.subplot(3,3,9)
plt.title('Quakes in Papua New Guinea by Year: Barplot', pad=20, fontsize=16)
ax = sns.countplot(x='Year', data=df_country[df_country['country'] == "Papua New Guinea"], color='#D2B48C', alpha=0.3)

for p in ax.patches: 
        ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+ p.get_width()/2, p.get_height() + 0.1), ha='center', va='bottom')
plt.ylabel(None), plt.yticks([]), plt.xticks(fontsize=10, rotation=90), plt.xlabel(None)

plt.show()

fig = px.density_mapbox(df_country[df_country['country'] == "Indonesia"], lat='latitude', lon='longitude', z='magnitude', radius=10,
                        center=dict(lat=0, lon=120), zoom=2.6, title = "Quakes in Indonesia by Latitude & Longitude")
fig.update_layout(mapbox_style="stamen-terrain", title_font_size=16, title_x=0.5)
fig.show()

fig = px.density_mapbox(df_country[df_country['country'] == "Japan"], lat='latitude', lon='longitude', z='magnitude', radius=10,
                        center=dict(lat=34, lon=145), zoom=3, title = "Quakes in Japan by Latitude & Longitude")
fig.update_layout(mapbox_style="stamen-terrain", title_font_size=16, title_x=0.5)
fig.show()

fig = px.density_mapbox(df_country[df_country['country'] == "Papua New Guinea"], lat='latitude', lon='longitude', z='magnitude', radius=10,
                        center=dict(lat=-6, lon=150), zoom=3.5, title = "Quakes in Papua New Guinea by Latitude & Longitude")
fig.update_layout(mapbox_style="stamen-terrain", title_font_size=16, title_x=0.5)

fig.show()


# In[25]:


## We will conclude the notebook by visualising the worldwide earthquake events by latitude and longitude

fig = px.density_mapbox(df_country, lat='latitude', lon='longitude', z='magnitude', radius=6,
                        center=dict(lat=0, lon=200), zoom=0, mapbox_style="stamen-terrain")
fig.update_layout(autosize=False, width=760, height=450,showlegend=False, title='Quakes by Country - Density Mapbox', title_x=0.5)

fig.show()

