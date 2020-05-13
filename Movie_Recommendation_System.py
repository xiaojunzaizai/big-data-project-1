#!/usr/bin/env python
# coding: utf-8
'''
CSC 6760 Big Data
Jun Chen
Movie Recommendation System
'''

# In[1]:


from pyspark.context import SparkContext
from pyspark.mllib.recommendation import ALS
from pyspark import SparkConf
from pyspark.sql import SparkSession, SQLContext, Row
from pyspark.sql.functions import *
import math
from pyspark.sql.types import IntegerType
import time


# In[2]:


def get_time(t):
    timeArray = time.localtime(t)
    otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
    return otherStyleTime


# In[3]:


def time_cov(t):
    timeArray = time.strptime(t, "%Y-%m-%d %H:%M:%S")
    timestamp = time.mktime(timeArray)
    return timestamp


# In[4]:


def change_value(v):
    if v >4.0 or math.isclose(v,4.0):
        return 2
    if v >3.0 and v < 4.0:
        return 1
    if v >2.0 and v <3.0 or math.isclose(v,2.0) or math.isclose(v,3.0):
        return 0
    if v>1.0 and v< 2.0:
        return -1
    if v >0.0 and v<1.0 or math.isclose(v,0.0) or math.isclose(v,1.0):
        return -2


# In[5]:


conf = SparkConf().setMaster("local") \
                  .set("spark.executor.memory", "4g") 
sc = SparkContext(conf= conf)
sc.setLogLevel("ERROR")

spark = SparkSession \
    .builder \
    .appName("Recommendation") \
    .config(conf= conf) \
    .getOrCreate()
# In[6]:


ratingdata = sc.textFile('ratings.csv')
ratingdata = ratingdata.filter(lambda x: 'movieId' not in x)
moviedf= spark.read.csv('movies.csv')
ratingdata.persist()
ratingdf = spark.read.csv('ratings.csv')


# In[7]:


ratingdata.take(5)


# In[8]:


ratingrdd = ratingdf.rdd.map(lambda attributes: \
                            Row(userId = attributes[0],\
                                movieId = attributes[1],\
                                rating = attributes[2],\
                                timestamp = attributes[3]))

# In[9]:


ratingdf1 = ratingrdd.toDF()
ratingdf1 = ratingdf1.filter(ratingdf1.movieId != 'movieId')
ratingdf1 = ratingdf1.withColumn('timestamp',ratingdf1['timestamp'].cast(IntegerType()))
ratingdf1.show()


# In[10]:


ratingdf1.describe().show()


# In[11]:


max = int(ratingdf1.describe('timestamp').filter("summary = 'max'").select('timestamp').collect()[0].asDict()['timestamp'])
print('MAX timestamp: '+ str(max))
min = int(ratingdf1.describe('timestamp').filter("summary = 'min'").select('timestamp').collect()[0].asDict()['timestamp'])
print('Min timestamp: '+ str(min))


# In[12]:


moviedf.show()


# In[13]:


# movie dictionary
movierdd = moviedf.rdd.map(lambda attributes: \
                            Row(movieId = attributes[0],\
                            title = attributes[1]))
Df1 = movierdd.toDF()

result = Df1.filter(Df1.movieId != 'movieId')
moviedict = result.rdd.map(lambda row: {row[0]: row[1]}).collect()

movietitle={}
for i in moviedict:
    movietitle.update(i)


# In[14]:


print(movietitle['1'])


# In[15]:


# print(movietitle)


# In[16]:


rating_num = ratingdata.count()


# In[17]:


print('total data in rating: ',end = '')
print(rating_num)


# In[18]:


movierating = ratingdata.map(lambda x: x.split(',')[:3])
movierating.persist()


# In[19]:


movierating.take(5)


# In[20]:


total_user = movierating.map(lambda x: x[0]).distinct().count()
total_movie = movierating.map(lambda x: x[1]).distinct().count()
movienum=result.distinct().count()


# In[21]:


print('total data in rating: ',end = '')
print(rating_num)
print('total user number in rating: ',end = '')
print(total_user)
print('total movie number in rating: ', end = '')
print(total_movie)
print('total movie number: ',end = '')
print(movienum)


# In[22]:


#use ALS to train the dataset
start = time.time()
model = ALS.train(movierating,10,10,0.01)
stop = time.time()


# In[23]:


print('Training time: ', str(stop-start))


# In[24]:


# recommendation based on ALS
while True: 
    userID = int(input('Type a userID from 1 to 610  '))
    if userID>=1 and userID<=610:
        break
r = model.recommendProducts(userID,5)
print('For User', str(r[0][0]))
print('Recommendation movies: ' )
for u in r:
    print('   ',movietitle[str(u[1])], 'Like Rate: ', str(u[2]))


# In[25]:


# whole dataset top 10 most people watched movies
movie_fq = movierating.map(lambda x: (x[1],1)).reduceByKey(lambda x,y: x+y)
movie_fq_sort = movie_fq.sortBy(lambda x: x[1],False).map(lambda x:x[0])

most10 = movie_fq_sort.take(10)
print('Most people watch: ')
for i in most10:
    print(movietitle[i])


# In[26]:


# whole dataset top 10 high rating movies
movierating1 = movierating.map(lambda x: (x[1],float(x[2]))).mapValues(change_value).reduceByKey(lambda x,y: x+y).sortBy(lambda x: x[1],False).map(lambda x: x[0])

top10 = movierating1.take(10)
print('Top 10 High Rating Movie: ')
for i in top10:
    print(movietitle[i])


# In[27]:


# get 10 year, 5 yeay and 2 year timestamp
print(get_time(min))
print(get_time(max))
maxtime1 = get_time(max).split(' ')
maxtime2 = maxtime1[0].split('-')
year = int(maxtime2[0])
maxtime3 = '-'.join(maxtime2) + ' '+maxtime1[1]
tenyear1 = str(year -10)
maxtime2[0] = tenyear1
tenyear2 = '-'.join(maxtime2) + ' '+maxtime1[1]
tenyear = int(time_cov(tenyear2))
print('Ten year timestamp: '+str(tenyear))
fiveyear1 = str(year -5)
maxtime2[0] = fiveyear1
fiveyear2 = '-'.join(maxtime2) + ' '+maxtime1[1]
fiveyear = int(time_cov(fiveyear2))
print('Five year timestamp: '+str(fiveyear))
twoyear1 = str(year -2)
maxtime2[0] = twoyear1
twoyear2 = '-'.join(maxtime2) + ' '+maxtime1[1]
twoyear = int(time_cov(twoyear2))
print('Two year timestamp: '+str(twoyear))


# In[28]:


movietime = ratingdata.map(lambda x: x.split(',')[1:])
movietime.persist()


# In[29]:


movietime.take(3)


# In[30]:



moveitime1 = movietime.filter(lambda x: int(x[2])>=tenyear)
moveitime2 = movietime.filter(lambda x: int(x[2])>=fiveyear)
moveitime3 = movietime.filter(lambda x: int(x[2])>=twoyear)
moveitime1.persist()
moveitime2.persist()
moveitime3.persist()


# In[31]:


moveitime1.take(3)


# In[32]:


#ten year
movie_ten = moveitime1.map(lambda x: (x[0],1)).reduceByKey(lambda x,y: x+y).sortBy(lambda x: x[1],False).map(lambda x:x[0])

tenyear_most10 = movie_ten.take(10)
print('In Ten Years \nMost people watch: ')
for i in tenyear_most10:
    print(movietitle[i])
print()

ten_top = moveitime1.map(lambda x: (x[0],float(x[1]))).mapValues(change_value).reduceByKey(lambda x,y: x+y).sortBy(lambda x: x[1],False).map(lambda x: x[0])

tenyear_top10 = ten_top.take(10)
print('In Ten Year \nTop 10 High Rating Movie: ')
for i in tenyear_top10:
    print(movietitle[i])


# In[33]:


#five years
movie_five = moveitime2.map(lambda x: (x[0],1)).reduceByKey(lambda x,y: x+y).sortBy(lambda x: x[1],False).map(lambda x:x[0])

fiveyear_most10 = movie_five.take(10)
print('In Five Years \nMost people watch: ')
for i in fiveyear_most10:
    print(movietitle[i])
print()

five_top = moveitime2.map(lambda x: (x[0],float(x[1]))).mapValues(change_value).reduceByKey(lambda x,y: x+y).sortBy(lambda x: x[1],False).map(lambda x: x[0])

fiveyear_top10 = five_top.take(10)
print('In Five Year \nTop 10 High Rating Movie: ')
for i in fiveyear_top10:
    print(movietitle[i])


# In[34]:


#two years
movie_two = moveitime3.map(lambda x: (x[0],1)).reduceByKey(lambda x,y: x+y).sortBy(lambda x: x[1],False).map(lambda x:x[0])

twoyear_most10 = movie_two.take(10)
print('In Two Years \nMost people watch: ')
for i in twoyear_most10:
    print(movietitle[i])
print()

two_top = moveitime3.map(lambda x: (x[0],float(x[1]))).mapValues(change_value).reduceByKey(lambda x,y: x+y).sortBy(lambda x: x[1],False).map(lambda x: x[0])

twoyear_top10 = two_top.take(10)
print('In Two Year \nTop 10 High Rating Movie: ')
for i in twoyear_top10:
    print(movietitle[i])

