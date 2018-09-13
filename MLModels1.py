#importing libs
import warnings, datetime, csv,itertools, pandas as pd, matplotlib.pyplot as plt
import numpy as np
# for csv data 
from pandas import DataFrame
from pandas import concat
from pandas import Series 
# for normalization and scaling
from sklearn import preprocessing
# for measuring the accuracy of the model
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from keras import metrics
# models for fitting and predicting
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
# for correlation plotting
from pandas.tools.plotting import lag_plot
# for preventing warnings messages
warnings.filterwarnings('ignore')

#reading the dataset and converting date & time column into datetime obj
df = pd.read_csv('/home/reem/Desktop/codes/WQ/wqfull1.csv')
df['datetime'] =  pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M')
colors = ['r','b','y','g','c', 'm']
models_names = ['MLP','Linear Regression','Logistic Regression','Decision Tree Regressor','Deep Neural Networks','Autoregresstion']
features_names = ['Sampling depth','Temperature','pH','Chlorophylls']
accuracy = [explained_variance_score,mean_absolute_error,mean_squared_error,mean_squared_log_error,r2_score]

class MLModels():
    
    def data_clean_split(feature):
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
        
        #splitting the data into training and testing
        stop = int(len(date_measurements)*0.6)
        x_train = inp[:stop,:]
        y_train = out[:stop,:]
        x_test = inp[stop:stop+int(stop/3),:]
        y_test = out[stop:stop+int(stop/3),:]
        return(x_train, y_train, x_test, y_test,date_measurements.flatten())
    
    def MLP_model(x_train,y_train,x_test,y_test):
        #try different activation functions with different solvers and different layers to find the best one then pass it 
        mlp_mse = []
        mlp_y = []
        for i in (['identity','logistic', 'tanh', 'relu']):
            for j in (['lbfgs', 'sgd', 'adam']):
                for k in range(1,11):
                    if (i=='identity') & (j=='sgd') | (i=='relu') & (j=='sgd'):
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
        return(mlp_y[mlp_mse_ind[0][0]])

    def LR_model(x_train,y_train,x_test):
        linear = linear_model.LinearRegression()
        linear.fit(x_train, y_train)
        return(linear.predict(x_test))
    
    def MOR_model(x_train,y_train,x_test):
        MOR = MultiOutputRegressor(GradientBoostingRegressor(random_state=0))
        MOR = MOR.fit(x_train, y_train)
        return(MOR.predict(x_test))
    
    def DTR_model(x_train,y_train,x_test):
        tree = DecisionTreeRegressor()
        tree = tree.fit(x_train, y_train)
        return(tree.predict(x_test))
    
    def baseline_model():
            model = Sequential()
            model.add(Dense(12, input_dim=18, kernel_initializer='normal', activation='relu'))
            model.add(Dense(8, kernel_initializer='normal', activation='relu'))
            model.add(Dense(6, kernel_initializer='normal'))
            model.compile(loss='mean_squared_error',optimizer='adam',metrics=[metrics.mse])
            return model

    def DNN_model(x_train,y_train,x_test,model):
        estimator = KerasRegressor(build_fn=model, epochs=20, batch_size=5, verbose=0)
        history = estimator.fit( x_train, y_train)
        return(estimator.predict(x_test))
    
    def AR_model(date_measurements):
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
        return(test_y1,predictions)
        
    def plot_model(y_test,y_test_AR,y_predict):
        #--- Predicted vs Actual for 6 models ---
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
        for i in range(len(accuracy)):
            print(accuracy[i])
            for j in range(len(y_predict)):
                if j == 5:
                    print('%-30s'%models_names[j],'\t',accuracy[i](y_test_AR, y_predict[j]))
                else:
                    print('%-30s'%models_names[j],'\t',accuracy[i](y_test, y_predict[j]))
            print('\n')



