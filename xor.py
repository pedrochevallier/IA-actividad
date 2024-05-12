import numpy as np
from keras.models import Sequential
from keras.layers import Dense

'''
xor(0,0) = 0
xor(0,1) = 1
xor(1,0) = 1
xor(1,1) = 0
'''

train_data = np.array([[0,0],[0,1],[1,0],[1,1]], "float32")
target_data = np.array([[0],[1],[1],[0]],"float32")

model = Sequential()
model.add(Dense(16, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['binary_accuracy'])

print(model.summary())

model.fit(train_data, target_data, epochs=5000)

scores = model.evaluate(train_data, target_data)
print(scores)

print('*' * 50)
print(model.predict(train_data))