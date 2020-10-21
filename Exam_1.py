
# coding: utf-8

# In[1]:


import pandas as pd
import xml.etree.cElementTree as et
import os
import multiprocessing as mp
import pandas.util.testing as pdt
import time
import numpy as np
from datetime import datetime as dt
from datetime import date
import pygal
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
timerstart = time.time()


# # 1a

# In[2]:


def multip(filename):
    file = os.path.join(path, filename)
    doc = et.parse(file)
    inner_data = pd.DataFrame()
    fields =  ['MRN', 'LRN', 'ISVOC', 'UniqueNumber', 'CustomsOffice', 'CpcCode', 'PreviousCpc',
               'TransportCode', 'CountryExport','CountryImport', 'TotalEntryLines', 'TotalWeight', 
               'TotalPackages', 'ServiceProvider', 'SPSource', 'HSCode', 'HasPermit',
               'CustomsValue', 'TradeAgreement', 'StatsQuantity', 'StatsUOM', 'CountryOfOrigin', 
               'DateSubmitted', 'DateControlReceived', 'CustomsResponses']
    cusfields = ['StatusCode', 'MessageText', 'DateTimeReceived']
    for elem in doc.findall('.//BOEHeader'):    
       # inner_dict = pd.Series({k:None for k in fields})   # PRE-POPULATES TEMP DICT
        inner_dict = pd.Series()
        df = pd.DataFrame(columns=fields)        
        i = 0
        for item in elem.findall('.//*'):        
            if item.tag in fields:
                inner_dict[item.tag] = item.text
                if item.tag == 'DateSubmitted':
                    if item.text == '0000-00-00-00.00.00':
                        inner_dict[item.tag] = None
                    else:
                        inner_dict[item.tag] = dt.strptime(item.text, '%Y-%m-%d-%H.%M.%S')
                if item.tag == 'DateControlReceived':
                    if item.text == '0000-00-00-00.00.00':
                        inner_dict[item.tag] = None
                    else:
                        inner_dict[item.tag] = dt.strptime(item.text, '%Y-%m-%d-%H.%M.%S')
                if item.tag == 'CustomsResponses':                      
                    cus_data = pd.DataFrame(columns = cusfields)
                    for cus in item.findall('.//CustomsResponse'):                         
                        cus_dict = pd.Series()
                        for res in cus.findall('.//*'):
                            cus_dict[res.tag] = res.text
                            
                            if res.tag == 'StatusCode':                                
                                cus_dict[res.tag] = int(res.text)
                            if res.tag == 'DateTimeReceived':
                                if item.text == '0000-00-00-00.00.00':
                                    cus_dict[res.tag] = None
                                else:
                                    cus_dict[res.tag] = dt.strptime(res.text, '%Y-%m-%d-%H.%M.%S')
                        cus_data = cus_data.append(cus_dict, ignore_index=True)
                    inner_dict['CustomsResponses'] = cus_data.sort_values(['DateTimeReceived'])  
                            
        inner_data = inner_data.append(inner_dict, ignore_index=True)
    #print(file)
    return inner_data


# In[3]:


path = "/media/werner/Google Drive/Google Drive/University/Year 4/REII424 - Data Analysis/Exam 1/Small/"
xml_data = pd.DataFrame()
    
if __name__ == '__main__':
    filenames = os.listdir(path)
    p = mp.Pool(processes = 8)
    start = time.time()    
    splitfiles = np.array_split(filenames,8)
    async_result = p.map(multip, filenames)
    xml_data = xml_data.append(async_result, ignore_index=True, sort=True)
    p.close()
    p.join()
    end = time.time()
    m, s = divmod(end-start, 60)
    h, m = divmod(m, 60)
    print("Time = %d:%02d:%f"%(h,m,s))
    
print("Number of files = %d"%len(filenames))
print("Number of entries = %d"%len(xml_data))


# In[4]:


xml_data = xml_data[~xml_data['DateSubmitted'].isnull()]
xml_data = xml_data[~xml_data['DateControlReceived'].isnull()]


# In[5]:


xml_data.head()


# In[6]:


xml_data['CustomsResponses'][0]


# # 1b

# In[7]:


xml_data['NoCustomsResponses'] = (xml_data['CustomsResponses']).apply(len)
xml_data.head()


# In[8]:


len(xml_data)


# In[9]:


pd.DataFrame(xml_data['NoCustomsResponses']).head(10)


# # 1c

# In[10]:


xml_data.sort_values(['DateSubmitted'])
xml_data[['UniqueNumber', 'DateSubmitted']].head(20)


# # 1d

# In[11]:


def category(code):
    var = int(code[0:2]) if isinstance(code, str) else None
    values = [-1, 5, 15, 24, 27, 38, 40, 43, 49, 63, 67, 71, 83, 85, 89, 97, 99]
    cat = ['Animal', 'Vegetable', 'Food',  'Mineral', 'Chemical', 'Plastic',
           'Hide', 'Wood', 'Textile', 'Footwear', 'Stone and Glass', 'Metal', 
          'Machinery', 'Transport', 'Miscellaneous', 'Service']
    dict = {}
    for i in range(len(cat)):
        dict.update(dict.fromkeys(list(range(values[i]+1, values[i+1]+1)), cat[i]))

    return dict.get(var,"Other")

HS = xml_data['HSCode']
HS = HS.str.strip()
HS[HS == ""] = None
cat = []

start= time.time()    
xml_data['Category'] = HS.apply(category)
end= time.time()    
m, s = divmod(end-start, 60)
h, m = divmod(m, 60)
print("Time = %d:%02d:%f"%(h,m,s))
xml_data[['UniqueNumber', 'DateSubmitted', 'HSCode', 'Category']].head(20)


# # 1e

# In[12]:


duration = pd.DataFrame()
firstdate = pd.DataFrame()
lastdate = pd.DataFrame()
for index, row in xml_data.iterrows():
    temp = row['CustomsResponses']['DateTimeReceived'].max()
    if temp == np.nan:
        temp = row['DateControlReceived']   
    lastdate = lastdate.append([temp], ignore_index=True)
    firstdate = firstdate.append([row['DateSubmitted']], ignore_index=True)
    
xml_data['FirstDate'] = firstdate    
xml_data['LastDate'] = lastdate
for index, row in xml_data.iterrows():
    duration = duration.append([row['LastDate']-row['FirstDate']], ignore_index=True)
    
duration
xml_data['Duration'] = duration
xml_data = xml_data[~xml_data['Duration'].isnull()]


# In[13]:


statuscodes = pd.DataFrame()
not_stopped = pd.DataFrame()
stopped = pd.DataFrame()
inspected = pd.DataFrame()
amended = pd.DataFrame()
months = pd.DataFrame()
requested = pd.DataFrame()    
notstopcodes = set([2, 4, 13, 31, 36])
stopcodes = set([2])
inspectcodes = set([36])
amendcodes = set([6, 26])
requestedcodes = set([13, 31])


def processcodes(row):    
    codes = set(row['CustomsResponses']['StatusCode'])
    row['Codes'] = codes
    row['Stopped'] = not not codes & stopcodes
    row['Not Stopped'] = not codes & notstopcodes
    row['Inspected'] = not not codes & inspectcodes
    row['Amended'] = not not codes & amendcodes
    row['Requested'] = not not codes & requestedcodes
    row['Month'] = date.strftime(row['FirstDate'],"%Y %m")
    return(row)

def process(df):
    res = df.apply(processcodes, axis=1)
    return res

start= time.time()  

if __name__ == '__main__':
    p = mp.Pool(processes=8)
    split_dfs = np.array_split(xml_data,8)
    pool_results = p.map(process, split_dfs)
    p.close()
    p.join()

    # merging parts processed by different processes
    xml_data = pd.concat(pool_results, axis=0)


end= time.time()    
m, s = divmod(end-start, 60)
h, m = divmod(m, 60)
print("Time = %d:%02d:%f"%(h,m,s))


# In[14]:


xml_data = xml_data[~xml_data['Codes'].isnull()]
xml_data[['UniqueNumber', 'DateSubmitted', 'Codes', 'Not Stopped', 'Stopped', 'Inspected', 'Amended']].head(20)


# # 1f

# In[15]:


categories = ['CustomsOffice', 'CpcCode', 'PreviousCpc', 'CountryOfOrigin', 
              'CountryExport', 'CountryImport', 'TransportCode', 'SPSource', 'HSCode']
cat_count = {}
cat_amount = pd.DataFrame(columns=['Categories'])
for category in categories:    
    df = pd.DataFrame()
    df['Transactions'] = xml_data.groupby([category])[category].count()
    cat_amount.loc[category] = xml_data.groupby([category])[category].count().count()
    cat_count["{0}".format(category)] = df


# In[16]:


cat_amount


# # 2a,b,c

# In[17]:


cat_count_month = {}
cat_count_month_req = {}
cat_count_month_frac = {}
for category in categories:    
    monthly = xml_data.groupby([category, 'Month']).size().reset_index().pivot(values=0, index='Month', columns=category)
    monthly_req = xml_data[xml_data['Requested'] == True].groupby([category, 'Month']).size().reset_index().pivot(values=0, index='Month', columns=category)
    monthly_frac = (monthly_req/monthly)
    cat_count_month["{0}".format(category)] = monthly
    cat_count_month_req["{0}".format(category)] = monthly_req
    cat_count_month_frac["{0}".format(category)] = monthly_frac

cat_count_month = pd.concat(cat_count_month.values(), axis=1, keys=cat_count_month.keys(), sort=True)
cat_count_month_req = pd.concat(cat_count_month_req.values(), axis=1, keys=cat_count_month_req.keys(), sort=True)
cat_count_month_frac = pd.concat(cat_count_month_frac.values(), axis=1, keys=cat_count_month_frac.keys(), sort=True)
cat_count_month_frac_acc = cat_count_month_frac.rolling(100, min_periods=1).mean()


# # 2d

# In[18]:


def getval(row):
    new = pd.Series()
    for category in categories:
        if not row[category] == None and not row[category] == np.nan and not row['DateSubmitted'] == None:
            new["val_{0}".format(category)] = (cat_count_month_frac_acc[category][row[category]][date.strftime(row['DateSubmitted'],"%Y %m")])
        else:
            new["val_{0}".format(category)] = 0
    return(new)


def process(df):
    res = df.apply(getval, axis=1)
    return res

  
val_names = list("val_{0}".format(category) for category in categories)  
start= time.time()
if __name__ == '__main__':
    p = mp.Pool(processes=8)
    split_dfs = np.array_split(xml_data,8)
    pool_results = p.map(process, split_dfs)
    p.close()
    p.join()

    # merging parts processed by different processes
    cat_val = pd.concat(pool_results, axis=0)
#cat_val = pd.DataFrame(columns=val_names)
#for index, row in xml_data.iterrows():
#    temp = pd.Series(index=val_names)
#    for category in categories:
#        if not row[category] == None:
#            temp["val_{0}".format(category)] = (cat_count_month_frac_acc[category][row[category]][date.strftime(row['DateSubmitted'],"%Y %m")])
#        else:
#            temp["val_{0}".format(category)] = 0
            
#    cat_val = cat_val.append(temp, ignore_index=True)
data_with_val = pd.concat([xml_data, cat_val], axis=1)
end= time.time()    
m, s = divmod(end-start, 60)
h, m = divmod(m, 60)
print("Time = %d:%02d:%f"%(h,m,s))


# In[19]:


data_with_val


# In[20]:


cat_val.head(10)


# # 2e

# In[21]:


data_with_val = data_with_val[~data_with_val['Requested'].isnull()]
data_with_val['Requested'] = data_with_val['Requested'].astype(int)


# In[22]:


corrcoeff = data_with_val[['Requested'] + val_names].fillna(0).corr().loc['Requested']
corrcoeff = pd.DataFrame(corrcoeff).reset_index().iloc[1:len(corrcoeff)]


# In[23]:


corrcoeff


# In[24]:


from pygal.style import CleanStyle
newstyle = CleanStyle(plot_background='transparent', font_family='googlefont:Source Sans Pro')


# In[25]:


chart = pygal.Bar(fill=True, style=newstyle,show_legend = False,x_label_rotation=-45, width=800, height=500)
chart.title = 'Pearson Correlation Coefficients Between Input Factors and Additional Documentation Requests'
chart.x_labels = categories
chart.x_title = 'Input Factor'
chart.y_title = 'Pearson Correlation Coefficient'
chart.add('', corrcoeff['Requested'])


# In[26]:


chart.render_to_png('/media/werner/Google Drive/Google Drive/University/Year 4/REII424 - Data Analysis/Exam 1/corrcoeff.png')


# # 3a,b

# In[27]:


import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
data = pd.concat([data_with_val['Requested'], cat_val], axis=1)
data.head()
data = data_with_val


# In[28]:


request = data[data['Requested'] == 1].sample(frac=1)


# In[29]:


norequest = data[data['Requested'] == 0]['Requested'].count()


# In[30]:


while data[data['Requested'] == 1]['Requested'].count() < norequest:
    data = data.append(request)
data = data.iloc[0:norequest*2]
data = data.reset_index(drop=True)


# In[31]:


X_train, X_test , y_train , y_test = train_test_split(data.loc[:, "val_CustomsOffice":"val_HSCode"],data['Requested'],test_size=0.3)
trainindexes = list(X_train.sort_index().index.values) 
traindata = data.iloc[trainindexes]


# # 3c

# In[32]:


corrcoeff = corrcoeff.sort_values('Requested', ascending=False)
highestcorr = list(corrcoeff['index'].iloc[0:3].str[4:100])


# In[34]:


for cat in highestcorr:
    temp = cat_count_month_frac[cat].reset_index().melt(id_vars=['index']).set_index(['index', 'variable']).sort_values(['index', 'variable']).dropna()
    temp2 = xml_data[['Requested', 'Month', cat]].groupby(['Month', cat]).count()
    temp3 = pd.concat([temp, temp2], axis=1)
    temp3.plot.scatter(x='value', y = 'Requested', title=cat)


# # 4a

# In[37]:


X_train = X_train.fillna(0)
X_test = X_test.fillna(0)
reg = LinearRegression().fit(X_train, y_train)
reg.score(X_train, y_train)
reg.coef_


# # 4b

# In[38]:


frac_cols = ['All Observations', 'Target One', 'Target Zero', 'Threshold']
df_frac = pd.DataFrame()

def process(i):
    threshold = i/100
    obs = pd.Series()
    y_pred = reg.predict(X_test)
    y_pred = y_pred > threshold
    y_pred = y_pred*1
    
    df = pd.DataFrame([y_pred, y_test], index=['Predicted', 'Target']).T
    df['Correct'] = df['Target'] == df['Predicted']
    obs_all = df.groupby(['Correct']).size().reset_index(name='Count')
    correct = int(obs_all.loc[obs_all['Correct'],'Count'])
    incorrect = int(obs_all.loc[~obs_all['Correct']]['Count'])
    obs['All Observations'] = correct/(correct+incorrect)
    obs_0 = df[df['Target'] == 0].groupby(['Correct']).size().reset_index(name='Count')

    correct = int(obs_0.loc[obs_0['Correct'],'Count'])
    incorrect = int(obs_0.loc[~obs_0['Correct']]['Count'])

    obs['Target One'] = correct/(correct+incorrect)
    obs_1 = df[df['Target'] == 1].groupby(['Correct']).size().reset_index(name='Count')

    correct = int(obs_1.loc[obs_1['Correct'],'Count'])
    incorrect = int(obs_1.loc[~obs_1['Correct']]['Count']) if not obs_1.loc[~obs_1['Correct']]['Count'].empty else 0

    obs['Target Zero'] = correct/(correct+incorrect)
    obs['Threshold'] = threshold
    return obs
    
if __name__ == '__main__':
    
    start = time.time()  
    p = mp.Pool(processes = 8)         
    async_result = p.map(process, [i for i in range(101)]) 
    df_frac = df_frac.append(async_result, ignore_index=True)         
    p.close()
    p.join()
    end = time.time()
    m, s = divmod(end-start, 60)
    h, m = divmod(m, 60)
    print("Time = %d:%02d:%f"%(h,m,s))
    
df_frac.head()


# In[40]:


chart = pygal.Line(style=newstyle,show_legend = True, width=800, height=500, show_dots=False, show_minor_x_labels=False, x_label_rotation=0.01, x_labels_major_count=11)
chart.title = 'Fraction Correct Predictions with Varying Threshold'
chart.x_labels = df_frac['Threshold']
chart.x_title = 'Threshold'
chart.y_title = 'Fraction Correct Predictions'
chart.add('All', df_frac['All Observations'])
chart.add('One', df_frac['Target One'])
chart.add('Zero', df_frac['Target Zero'])


# In[57]:


chart.render_to_png('/media/werner/Google Drive/Google Drive/University/Year 4/REII424 - Data Analysis/Exam 1/predict.png')


# # 4c

# In[41]:


df_frac.loc[(df_frac['Target One']-0.5).abs().argsort()[:1]]


# In[42]:


df_frac.loc[(df_frac['Target One']-0.8).abs().argsort()[:1]]


# In[43]:


df_frac.loc[(df_frac['Target One']-0.9).abs().argsort()[:1]]


# In[44]:


df_frac.loc[(df_frac['Target One']-0.95).abs().argsort()[:1]]


# In[45]:


timerend = time.time()
m, s = divmod(timerend-timerstart, 60)
h, m = divmod(m, 60)
print("Time = %d:%02d:%02d"%(h,m,s))


# In[55]:




