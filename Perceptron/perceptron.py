import numpy as np
import pandas as pd

df = pd.read_csv("placements.csv")
df.head()

# Extract features (X) and target (y)
X = df[['dsa_questions', 'cgpa']].values
y = df['placed'].values

# data normalization
X[:,0] = np.clip(X[:,0]/500,0,1)
# clip keeps the number in a given range
X[:,1] = X[:,1]/10

# initialize parameters
weights  = np.zeros(2)
bias = 0.0
learning_rate = 0.1
# learning rate decided senstivity of the weight changes 
epochs = 100


def activation(z):
    return 1 if z>=0 else 0

def predict(inputs):
    z = np.dot(inputs, weights) + bias
    return activation(z)

#training
def train(X_data, y_data):
  global weights, bias
  best_accuracy = -1.0
  best_weights = weights.copy()
  best_bias = bias

  for epoch in range(epochs):
    errors = 0 

    for inputs, target in zip(X_data,y_data):
      prediction = predict(inputs)
      error = target - prediction

      weights[:] += learning_rate * error * inputs
      bias += learning_rate * error

      if error != 0:
        errors += 1

    predictions = np.array([predict(row) for row in X_data])
    accuracy = np.mean(predictions == y_data)
    if accuracy > best_accuracy:
      best_accuracy = accuracy
      best_weights = weights.copy()
      best_bias = bias
      
    print(f"Epoch { epoch +1}: Errors = {errors}")

    if errors == 0:
      print("Training converged!!")
      break

  weights[:] = best_weights
  bias = best_bias
  print(f"Best training accuracy: {best_accuracy * 100:.2f}%")

def evaluate(X_data,y_data):
  predictions = np.array([predict(row) for row in X_data])
  accuracy = np.mean(predictions == y_data) * 100
  print(f"Accuracy: {accuracy: .2f}%")

train(X, y)
evaluate(X, y)

dsa = float(input("Enter DSA marks: "))
cgpa = float(input("Enter CGPA : "))

student = np.array([min(dsa/500,1),
                    cgpa/10])

result = predict(student)

if result == 1:
  print("Student will get placed")
else:
  print("Student will not get placed")