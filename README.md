## MODEL 1

### Target :- 

- Take the model from last week, and observe the maximum validation accuracy obtained after reducing  the number of kernels in the middle layers to bring down the parameter count to around 8000.
- Run the experiment with the same optimizer as last week, but without rate schedulers.

### Result

- Total number of parameters = 8,162
- Best Train and Validation accuracy = 98.79% (14th epoch) and 99.34% (9th epoch) respectively

### Analysis

- Naively reducing the number of kernels from the previous submission does not work. I'm repeating a bunch of 3 * 3 Convolution layers after the first layer which stack up the parameter count without increasing the number of kernels available for feature extraction. Experimenting with the number of Maxpool and Conv2d layers might help here. 
- The gap between training and validation accuracy suggests that the model is underfitting. Playing around with the dropout rates might help reduce the gap between the two.
- The validation accuracy does not improve consistently with the number of epochs. It peaks at 99.34% and then drops down again and finally settles at 99.02 in the final epoch. The model performance can be finetuned with varying the optimizer and learning rates, which will be experimented with later.

## MODEL 2 

### Target :- 

- Vary the number of Conv2d / Maxpool layers to bring the parameter count to under 8000, while retaining the model performance of the first model. 
- Reduce model underfitting by playing around with the dropout rates. 


### Result

- Total number of parameters = 7,384
- Best Train and Validation accuracy = 99.16% (14th epoch) and 99.34% (11th epoch) respectively

### Analysis

- The maximum validation accuracy, while still lower than the benchmark, is closer to what was achieved in the previous experiment, but with # parameters < 8000. 
- The gap between the training and validation accuracy has narrowed down after experimenting with dropout rates, as compared to the previous model. 
- The validation accuracy jumps around quite a bit during the final epoches, which suggests that the model training might benefit from a lower learning rate during the later epochs. Experimenting with learning rate schedulers should smoothen the training process as well inch closer to hitting the benchmark of 99.4%


## MODEL 3


### Target:
- Experiment with Learning Rate Schedulers to reduce variation in model validation towards the latter epoches. 
- Hit the validation benchmark of 99.4% consistently during the final training epoches. 
    
### Result

- Total number of parameters = 7,384
- Best Train and Validation accuracy = 99.56% (14th epoch) and 99.5% (15th epoch) respectively

### Analysis

- The model hits the benchmark in the 8th epoch, and remains above it till the training ends, indicating the model might improve further with additional training.
- The gap between training and validation accuracy improves over the previous model.



