![alt text](https://upload.wikimedia.org/wikipedia/commons/b/b9/The_Icebergs_%28Frederic_Edwin_Church%29%2C_1861_%28color%29.jpg)


step 0: change pytorch version in environment, I have 
to use a special dist of pytorch because Apple hates me

this creates the environment with the needed dependencies
```conda env create -f environment.yml```

currently running with task0.yaml
```python train.py --config config/{task_name}```     


