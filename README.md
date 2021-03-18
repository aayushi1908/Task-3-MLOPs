# Task-3-MLOPs

ML integration with DevOps - Current need of the industry!!
A time consuming task while training the Machine Learning models is to continuously tweak the Hyper-Parameters to reach our desired Accuracy. It is one of the reasons why most of the ML related projects fail. 
This can be resolved upto an extent with  MLOPs = ML+ DevOps


In this blog, I'm explaining my MLOPs project which trains and tweaks a CNN model for Cat and Dog prediction from the dataset. My project uses "Jenkins" as an automation-tool and "GitHub" where the developer pushes the code.
Requirements for setting up the project :
1. Git
2. Jenkins
3. Redhat 8 VM
4. Docker

Project :
Creating Environments :
I've created 3 environments (images) in Docker using Dockerfile for running my programs -
1) env1 - This environment is for running any basic program which uses numpy and pandas.
To run the container of env1 -
docker run -it --name con_Basic env1



2) env2 - This environment is for running Old ML programs which use sklearn.
To run the container of env2 -
docker run -it --name con_ML env2


3) env3 -  This environment is for runnig DL programs which use keras.
To run the container of env3 -
docker run -it --name con_DL env3


Build Pipeline : This is the build pipeline for my chain of jobs. 


Making the model :
I've made a Cat and Dog prediction CNN model which uses the concept of Deep Learning.  This is a Binary Classification model.
Layers which I've used are-
-> Convolutional Layer
-> Max Pooling Layer
-> Flattering Layer
-> Dense Layers

I've uploaded the code on my GitHub.
Here's the link of my GitHub repository :

https://github.com/aayushi1908/Task-3-MLOPs.git

JENKINS JOBS :
I've made 8 jobs for this project-
JOB1 -- Pulling the code from GitHub whenever developer pushes or makes changes in the code.
This job pulls the code from GitHub and copies it to a folder named /Aayushi in Redhat.






JOB2 -- By looking at the program file, Jenkins automatically starts the respective container ( Eg - For CNN code, it should start a container of my "env3")




JOB3 -- Train the model and predict accuracy or metrics.




JOB4 -- This job finds if  the accuracy is > 80 % or not.
In this job, we check whether our obtained accuracy matches our desired accuracy and if  it matches, an email is sent. If doesn't, the code is tweaked.



JOB5 -- Notifies the user about the accuracy obtained by sending an Email.







For EMAIL CONFIGURATION -
For email configuration, set-
1. Set IMAP enable in your gmail account settings.
2. Set Permissions as ON for Less Secure apps.

JOB 6 -- For accuracy < 80 %,  tweak code runs.




JOB 7 -- After the tweak accuracy is obtained, an Email is sent to the Developer.





JOB 8 -- This is an extra job for Monitoring. If the container where code is running fails, this job starts it again automatically.


      
Tweak Code Console Output - 
This is the Console Output of my tweak code. Have a look.
















