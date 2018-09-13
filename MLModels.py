#--- importing required libraries ---
import warnings, datetime, csv,itertools, pandas as pd, matplotlib.pyplot as plt
import numpy as np

#--- for measuring the accuracy of the model ---
from keras import metrics
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score

#--- models used in prediction ---
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor

#--- To calculate computational time ---
import time

#--- For preventing warnings messages ---
warnings.filterwarnings('ignore')

#--- Reading the dataset and converting date & time column into datetime obj ---
df = pd.read_csv('/home/reem/Desktop/codes/WQ/wqfull.csv')
df['datetime'] =  pd.to_datetime(df['datetime'], format='%d/%m/%Y %H:%M')

#--- Define the variables to avoid redandancy ---
colors = ['r','b','y','g','c', 'm']
models_names = ['MLP','Linear Regression','Logistic Regression','Decision Tree Regressor','Deep Neural Networks','Autoregresstion']
features_names = ['Specific Conductance','Dissolved Oxygen','Chlorophyll','Turbidity']
accuracy = [explained_variance_score,mean_absolute_error,mean_squared_error,mean_squared_log_error,r2_score]
exe_time = []

#--- Class MLModels contain all the required functions ---
class MLModels():
    
    def data_clean_split(feature):
        '''
        @The function finds the days that contains less than 240 measurements per day (6 minutes interval) and delete 
        them, find dates that contains Nan values and delete them, split the rest into input (the first 18 hours measurements) and 
        output (the following 6 hours measurements), split the data into training and testing (60:40) values.
        @Input is the feature which is the WQ variable name
        @Output is the input,output training and testing values and the full measurement without splitting
        '''
        
        mylist = df.groupby([df['datetime'].dt.date])
        xx = mylist[feature].apply(list)

        date_measurements = [xx[i] for i in range(len(xx)) if len(xx[i])==240]

        date_measurements = np.array(date_measurements)

        inds = np.where(np.isnan(date_measurements))
        badind = list(set(inds[0]))

        date_measurements = np.delete(date_measurements, ([i for i in badind]), axis=0)
        
        inp = [date_measurements[i][j] for i in range(len(date_measurements)) for j in range(0,180,10)]  
        out = [date_measurements[i][j] for i in range(len(date_measurements)) for j in range(180,240,10)] 
        inp = np.array(inp)
        out = np.array(out)
        inp = inp.reshape(len(date_measurements),18)
        out = out.reshape(len(date_measurements),6)
        print(inp.shape,out.shape)
        
        stop = int(len(date_measurements)*0.6)
        x_train = inp[:stop,:]
        y_train = out[:stop,:]
        x_test = inp[stop:stop+int(stop/3),:]
        y_test = out[stop:stop+int(stop/3),:]
        return(x_train, y_train, x_test, y_test,date_measurements.flatten())
    
    def MLP_model(x_train,y_train,x_test,y_test):
        
        '''
        @The function finds the optimal parameters for the Multi-Layer Perceptron by trying differet activation functions and solvers
        @Input x_train, x_test: the first 18 hours measurements, y_train,y_test: the following 8 hours measurements in training and/
        testing data respectivly
        @Output is the solver, activation function, nodes in hidden layer and the MSE of the least MSEs 
        '''
            
        mlp_time_start = time.clock()
        #--- Try different activation functions with different solvers and different layers to find the best one then pass it ---
        mlp_mse = []
        mlp_y = []
        for i in (['identity','logistic', 'tanh', 'relu']):
            for j in (['lbfgs', 'sgd', 'adam']):
                for k in range(1,11):
                    if (i=='identity')&(j=='sgd'):
                        continue
                    else:
                        MLP = MLPRegressor(
                            hidden_layer_sizes=(k,),  activation=i, solver=j)
                        MLP = MLP.fit(x_train,y_train)
                        y = MLP.predict(x_test)
                        mlp_mse.append([i,j,k,mean_squared_error(y_test, y)])
                        mlp_y.append(y)
        mlp_mse = np.array(mlp_mse)
        mlp_mse_ind = np.where(mlp_mse[:,3] == str((min((mlp_mse[:,3]).astype(float)))))
        print(mlp_mse[mlp_mse_ind])
        mlp_time_elapsed = (time.clock() - mlp_time_start)
        exe_time.append(mlp_time_elapsed)
        return(mlp_y[mlp_mse_ind[0][0]])

    def LR_model(x_train,y_train,x_test):
        
        '''
        @The function finds the intercept and the slope that best fits the data
        @Input x_train, x_test: the first 18 hours measurements, y_test: the following 8 hours measurements in training data 
        @Output is the predicted measurements of the test set 
        '''
            
        linear_time_start = time.clock()
        linear = linear_model.LinearRegression()
        linear.fit(x_train, y_train)
        linear_time_elapsed = (time.clock() - linear_time_start)
        exe_time.append(linear_time_elapsed)
        return(linear.predict(x_test))
    
    def MOR_model(x_train,y_train,x_test):  
        
        '''
        @The function apply the Logistic Regressor into the training, testing data
        @Input x_train, x_test: the first 18 hours measurements, y_test: the following 8 hours measurements in training data 
        @Output is the predicted measurements of the test set 
        '''
        
        MOR_time_start = time.clock()
        MOR = MultiOutputRegressor(GradientBoostingRegressor(random_state=0))
        MOR = MOR.fit(x_train, y_train)
        MOR_time_elapsed = (time.clock() - MOR_time_start)
        exe_time.append(MOR_time_elapsed)
        return(MOR.predict(x_test))
    
    def DTR_model(x_train,y_train,x_test):        
        '''
        @The function apply the Decision Tree Regressor into the training, testing data
        @Input x_train, x_test: the first 18 hours measurements, y_test: the following 8 hours measurements in training data 
        @Output is the predicted measurements of the test set 
        '''
        
        DTR_time_start = time.clock()
        tree = DecisionTreeRegressor()
        tree = tree.fit(x_train, y_train)
        DTR_time_elapsed = (time.clock() - DTR_time_start)
        exe_time.append(DTR_time_elapsed)
        return(tree.predict(x_test))
    
    def baseline_model():
                
        '''
        @The function creates DNN model with the following number of nodes in the dense layer 18*8*6
        @Input Void
        @Output is the created model 
        '''
        
        model = Sequential()
        model.add(Dense(12, input_dim=18, kernel_initializer='normal', activation='relu'))
        model.add(Dense(8, kernel_initializer='normal', activation='relu'))
        model.add(Dense(6, kernel_initializer='normal'))
        model.compile(loss='mean_squared_error',optimizer='adam',metrics=[metrics.mse])
        return model

    def DNN_model(x_train,y_train,x_test,model):
                
        '''
        @The function apply DNN model into training, testing data
        @Input x_train, x_test: the first 18 hours measurements, y_test: the following 8 hours measurements in training data 
        @Output is the predicted measurements of the test set 
        '''
        
        DNN_time_start = time.clock()
        estimator = KerasRegressor(build_fn=model, epochs=20, batch_size=5, verbose=0)
        history = estimator.fit( x_train, y_train)
        DNN_time_elapsed = (time.clock() - DNN_time_start)
        exe_time.append(DNN_time_elapsed)
        return(estimator.predict(x_test))
    
    def AR_model(date_measurements): 
        
        '''
        @The function the regression function of the same variable in 1 step in time
        @Input the list of all measurements
        @Output predictions: the predicted values using Auto-regression, test_y1: the same dataset shifted by 1 step in the future
        '''
        
        AR_time_start = time.clock()
        date_measurements = DataFrame(date_measurements)
        dataframe1 = concat([date_measurements.shift(1), date_measurements], axis=1)
        dataframe1.columns = ['t-1', 't+1']
        X1 = dataframe1.values
        train1, test1 = X1[1:len(X1)-240], X1[len(X1)-240:]
        train_X1, train_y1 = train1[:,0], train1[:,1]
        test_X1, test_y1 = test1[:,0], test1[:,1]
        predictions = list()
        for x in test_X1:
            predictions.append(x)
        AR_time_elapsed = (time.clock() - AR_time_start)
        exe_time.append(AR_time_elapsed)
        return(test_y1,predictions)
        
    def plot_model(y_test,y_test_AR,y_predict):
        #--- Plotting Predicted vs Actual for 6 models ---
        for i in range(6):
            if i == 5:
                plt.subplot( 3, 2, i+1 )
                plt.plot(y_test_AR,colors[i])
                plt.plot(y_predict[i],"k-",alpha=0.4)
                plt.margins(0,0)
                plt.title(models_names[i])
                plt.grid()
            else:
                plt.subplot( 3, 2, i+1 )
                plt.plot(y_test,colors[i])
                plt.plot(y_predict[i],"k-")
                plt.margins(0,0)
                plt.title(models_names[i])
                plt.grid()
        plt.show()
        
    def acc_model(y_test,y_test_AR,y_predict):
        
        '''
        @The function finds the accuracy for the regression metrices
        @Input y_test,y_test_AR,y_predict: the actual value for Y, the predicted values from auto-regression and predicted values/
        from the auto-regression resectively
        @Output Void
        '''
      
        for i in range(len(accuracy)):
            print(accuracy[i])
            for j in range(len(y_predict)):
                if j == 5:
                    print('%-30s'%models_names[j],'\t',accuracy[i](y_test_AR, y_predict[j]))
                else:
                    print('%-30s'%models_names[j],'\t',accuracy[i](y_test, y_predict[j]))
            print('\n')
    def computational_time():
        return exe_time



