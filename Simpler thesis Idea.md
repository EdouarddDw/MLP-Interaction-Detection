things to try:
biggest question here is model archetecture. how big, how many epochs?



## base line

| BASE EXP     | base line | diffrent optimizer |
| ------------ | --------- | ------------------ |
| noise        | 0         | 0                  |
| optimizers   | adam      | SDG                |
| drop out     | 0         | 0                  |
| wieght decay | 0         | 0                  |
## noise level 0.2SD

| **Expermiment**  | **base EXP** | **base EXP** | **exp1** | **exp2** | **exp3** | **exp4** | **exp4** | **exp5** |
| ---------------- | -------- | -------- | ---- | ---- | ---- | ---- | ---- | ---- |
| **noise**        | 0.2      | 0.2      | 0.2  | 0.2  | 0.2  | 0.2  | 0.2  | 0.2  |
| **optimizer**    | adam     | adam     | SDG  | adam | adam | adam | SDG  | SDG  |
| **Dropout**      | 0        | 0        | 0    | 0.2  | 0    | 0.2  | 0.2  | 0.2  |
| **wieght decay** | 0        | 0        | 0    | 0    | L2   | L2   | 0    | L2   |
## Noise level 0.5SD

| **Expermiment**  | **base EXP** | **exp6** | **exp7** | **exp8** | **exp9** | **exp10** | **exp11** |
| ---------------- | ------------ | -------- | -------- | -------- | -------- | --------- | --------- |
| **noise**        | 0.5          | 0.5      | 0.5      | 0.5      | 0.5      | 0.5       | 0.5       |
| **optimizer**    | adam         | SDG      | adam     | adam     | adam     | SDG       | SDG       |
| **Dropout**      | 0            | 0        | 0.2      | 0        | 0.2      | 0.2       | 0.2       |
| **wieght decay** | 0            | 0        | 0        | L2       | L2       | 0         | L2        |

## noise level 1.0SD
| **Expermiment**  | **base EXP** | **exp12** | **exp13** | **exp14** | **exp15** | **exp16** | **exp17** |
| ---------------- | ------------ | --------- | --------- | --------- | --------- | --------- | --------- |
| **noise**        | 1.0          | 1.0       | 1.0       | 1.0       | 1.0       | 1.0       | 1.0       |
| **optimizer**    | adam         | SDG       | adam      | adam      | adam      | SDG       | SDG       |
| **Dropout**      | 0            | 0         | 0.2       | 0         | 0.2       | 0.2       | 0.2       |
| **wieght decay** | 0            | 0         | 0         | L2        | L2        | 0         | L2        |


maybe add noise on input as well
