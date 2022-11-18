 # cis511-project
 
## Stage 1: Simplicity Metrics
Goal: Implement 3+ Simplicity Metrics:
- [Flesch-Kincaid](https://stars.library.ucf.edu/cgi/viewcontent.cgi?article=1055&context=istlibrary) - Joseph 
- [BLEU](https://aclanthology.org/P02-1040.pdf) - Not taken yet
- [SARI](https://aclanthology.org/Q16-1029.pdf) - Ahmed
- [Lai's RNN/CNN](https://ojs.aaai.org/index.php/AAAI/article/view/9513/9372) - Not taken yet
- Or others! We have some freedome here to do whatever we want :)

### Procedure
1. Copy count_metric.py into a new file and rename to the metric to implement (eg. sari_metric.py).
2. Replace `CountMetric` with own metric class (eg. `SARI(SimplicityMetric)`)
   1. Implement metric computation in `__call__` and return the float representation of your metric
   2. If you need to train (for a learned metric), implement something in `__init__` for your metric class which takes the `tune_data` to train the model.
3. Run the file! The implemented `main()` function should print at least the first 20 metrics with `compute_metric_main`.

## Stage 2: DNN Simplifier
Stretch Goal
