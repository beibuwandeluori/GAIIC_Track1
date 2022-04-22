1. data structure:

```
data
  |---preliminary_testA.txt
  |---train
       |---train_*.txt
       |---attr_to_attrvals.json
```

2. code explain:


- `./src/`: utils code about optimizer, model, dataset, etc.

- `Solver.py`: main train code

- `Inferencer.py`: inference code

3. preprocess:

```
run data.ipynb
```

4. train full text model:

```
run Solver.py refering ./logs/baseline_text.yaml
```

5. train tag-wise model:

```
run Solver.py refering ./logs/baseline_tag.yaml
```

6. inference model:

```
run Inferencer.py
```