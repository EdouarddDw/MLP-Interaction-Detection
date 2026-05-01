```text
Mean AUROC:   0.61
Median AUROC: 0.55
Min AUROC:    0.00
Max AUROC:    1.00
```

| Function | Mean AUROC |
|---|---:|
| f1 | 0.536 |
| f2 | 0.609 |
| f3 | 0.451 |
| f4 | 0.453 |
| f5 | 0.748 |
| f6 | 0.543 |
| f7 | 0.639 |
| f8 | 0.825 |
| f9 | 0.501 |
| f10 | 0.828 |


| **Noise** | **Mean AUROC** |
| --------- | -------------- |
| 0.0       | 0.81           |
| 0.1       | 0.67           |
| 0.2       | 0.61           |
| 0.5       | 0.59           |
| 1.0       | 0.49           |


| **Setting**            | **Mean AUROC** |
| ---------------------- | -------------- |
| No regularisation      | 0.73           |
| Dropout only           | 0.62           |
| Weight decay only      | 0.54           |
| Dropout + weight decay | 0.48           |


| **Optimizer** | **Mean AUROC** |
| ------------- | -------------- |
| Adam          | 0.62           |
| SGD           | 0.60           |
