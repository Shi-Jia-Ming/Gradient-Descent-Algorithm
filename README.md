# README

## Gradient Descent Optimization for Stock Prediction

This Python script implements a form of gradient descent optimization for a stock prediction model. The script is divided into three main classes: `OutputLayer`, `InputLayer`, and `CalculateLayer`.

### OutputLayer

This class represents the output of the model, which is a vector of weights for each stock on each day. It includes methods for initializing the output layer, updating the weights, and checking if the optimization process has converged.

### InputLayer

This class represents the input to the model, which includes the actual and predicted returns for each stock on each day, as well as the covariance matrix of the returns. It includes methods for initializing the input layer and calculating the covariance matrix.

### CalculateLayer

This class performs the optimization, adjusting the weights in the output layer based on the gradient of the loss function with respect to the weights. It includes methods for performing a single step of the optimization process, calculating the gradient, and performing the full optimization process.

## Usage

The script reads in actual and predicted return data from Excel files, creates an `InputLayer` with this data, then creates a `CalculateLayer` with the `InputLayer`, and calls the `optimize` method to perform the gradient descent optimization. The optimized weights are then saved to a CSV file.

```python
# Read data
origin = pd.read_excel("./assets/origin.xlsx", engine="openpyxl", sheet_name="di^")
origin = np.array(origin)
origin = origin[:, 1280:]

# Read predicted data
predict = pd.read_excel("./assets/predict.xlsx", engine="openpyxl", sheet_name="y_pred")
predict = np.array(predict)
predict = predict[:, 1:]
predict = predict.T

# Create InputLayer and CalculateLayer
input_layer = InputLayer(origin, predict)
calc = CalculateLayer(input_layer)

# Perform optimization
calc.optimize()

# Save results
output_file = "./assets/output.csv"
np.savetxt(output_file, calc.output_layer.vector, delimiter=",", fmt="%.5f")
```

Please note that the paths to the input and output files, as well as the sheet names, may need to be adjusted based on your specific setup.