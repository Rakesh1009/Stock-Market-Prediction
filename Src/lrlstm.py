import numpy as np
import pandas as pd 
from pylab import mpl, plt
plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'


import math
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn

company  = "IBM"
x = ''

#data
if company == "Google":
    x = '2005'
elif company == "IBM":
    x = '1999'
dates = pd.date_range(x+'-11-23','2022-11-23',freq='B')
df1=pd.DataFrame(index=dates)
df_ibm=pd.read_csv("./../Input/"+company+".csv", parse_dates=True, index_col=0)
print(df_ibm.head())
df_ibm = df1.join(df_ibm)
df_ibm[['Close']].plot(figsize=(15,6))
plt.ylabel("stock_price")
plt.title(company)
plt.savefig("./../Output/close_"+company+".png")

#datacleaning
df_ibm=df_ibm[['Close']]
df_ibm.info()

df_ibm=df_ibm.fillna(method='ffill')
df_ibm.info()

#normalising
scaler = MinMaxScaler(feature_range=(-1, 1))
df_ibm['Close'] = scaler.fit_transform(df_ibm['Close'].values.reshape(-1,1))

# function to create train, test data given stock data and sequence length
def load_data(stock, look_back):
    data_raw = stock.values # convert to numpy array
    data = []
    
    # create all possible sequences of length look_back
    for index in range(len(data_raw) - look_back): 
        data.append(data_raw[index: index + look_back])
    
    data = np.array(data)
    print(data.shape)
    test_set_size = int(np.round(0.2*data.shape[0]))
    train_set_size = data.shape[0] - (test_set_size)
    
    x_train = data[:train_set_size,:-1,:]
    y_train = data[:train_set_size,-1,:]
    
    x_test = data[train_set_size:,:-1]
    y_test = data[train_set_size:,-1,:]
    
    return [x_train, y_train, x_test, y_test]

look_back = 60 # choose sequence length
x_train, y_train, x_test, y_test = load_data(df_ibm, look_back)
print('x_train.shape = ',x_train.shape)
print('y_train.shape = ',y_train.shape)
print('x_test.shape = ',x_test.shape)
print('y_test.shape = ',y_test.shape)

x_train = torch.from_numpy(x_train).type(torch.Tensor)
x_test = torch.from_numpy(x_test).type(torch.Tensor)
y_train = torch.from_numpy(y_train).type(torch.Tensor)
y_test = torch.from_numpy(y_test).type(torch.Tensor)

#model

input_dim = 1
hidden_dim = 32
num_layers =2
output_dim =1

class LRLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=32, num_layers=2, output_size=1, dropout=0.2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.linear_1 = nn.Linear(input_size, hidden_layer_size)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(hidden_layer_size, hidden_size=self.hidden_layer_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(num_layers*hidden_layer_size, output_size)

    def forward(self, x):
        batchsize = x.shape[0]

        # layer 1
        x = self.linear_1(x)
        x = self.relu(x)
        
        # LSTM layer
        lstm_out, (h_n, c_n) = self.lstm(x)

        # reshape output from hidden cell into [batch, features] for `linear_2`
        x = h_n.permute(1, 0, 2).reshape(batchsize, -1) 
        
        # layer 2
        x = self.dropout(x)
        #print(x.shape)
        predictions = self.linear_2(x)
        #print(predictions.shape)
        a = predictions[:, -1]
        #print(a.shape)
        return a#predictions[:,-1]

model = LRLSTM(input_size=input_dim, hidden_layer_size=hidden_dim,num_layers=num_layers, output_size=1,dropout=0.2)

loss_fn = torch.nn.MSELoss()

optimiser = torch.optim.Adam(model.parameters(), lr=0.1)

num_epochs = 200
hist = np.zeros(num_epochs)

# Number of steps to unroll
seq_dim =look_back-1  

for t in range(num_epochs):
    # Initialise hidden state
    # Don't do this if you want your LSTM to be stateful
    #model.hidden = model.init_hidden()
    
    # Forward pass
    y_train_pred = model(x_train)
    #print(y_train_pred.shape)

    loss = loss_fn(y_train_pred, y_train)
    if t % 10 == 0 and t !=0:
        print("Epoch ", t, "MSE: ", loss.item())
    hist[t] = loss.item()
    # Zero out gradient, else they will accumulate between epochs
    optimiser.zero_grad()

    # Backward pass
    loss.backward()

    # Update parameters
    optimiser.step()


y_test_pred = model(x_test)

#print(y_test_pred.shape)
#print(y_train_pred.shape)

y_train_pred = y_train_pred.unsqueeze(1)
y_test_pred = y_test_pred.unsqueeze(1)
#print(y_train_pred.shape)

# invert predictions
#y_train_pred.unsqueeze(1)
#y_test_pred.unsqueeze(1)
#y_train_pred.reshape(-1,1)	
y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
y_train = scaler.inverse_transform(y_train.detach().numpy())
y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
y_test = scaler.inverse_transform(y_test.detach().numpy())

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(y_train[:,0], y_train_pred[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(y_test[:,0], y_test_pred[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

figure, axes = plt.subplots(figsize=(15, 6))
axes.xaxis_date()

axes.plot(df_ibm[len(df_ibm)-len(y_test):].index, y_test, color = 'red', label = 'Real '+company+' Stock Price')
axes.plot(df_ibm[len(df_ibm)-len(y_test):].index, y_test_pred, color = 'blue', label = 'Predicted '+company+' Stock Price')
#axes.xticks(np.arange(0,394,50))
plt.title(company+' Stock Price Prediction Using LRLSTM')
plt.xlabel('Time')
plt.ylabel(company + ' Stock Price LRLSTM')
plt.legend()
plt.savefig('./../Output/'+company+'_pred_lrlstm.png')
plt.text(-5, 60, 'Train Score: %.2f RMSE' % (trainScore) , fontsize = 22)
plt.text(-5, 60, 'Test Score: %.2f RMSE' % (testScore) , fontsize = 22)
plt.show()


