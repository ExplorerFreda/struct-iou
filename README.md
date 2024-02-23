# Structured IoU for Evaluation of (Speech) Constituency Parsing

This library implements a metric (Struct-IoU) that compares two constituency parse trees, represented as (relaxed) segment trees, with structured tree alignment--based intersection over union ratio.

In contrast to existing metrics, Struct-IoU takes into account the word boundaries of speech constituency parse trees, and does not require an ASR system in the loop.
The metric can be easily extended to evaluate text constituency parsing, by assuming unit-length word boundaries.

See more details in the [paper](https://home.ttic.edu/~freda/data/papers/structaiou.pdf).

## Installation
```bash
git clone https://github.com/ExplorerFreda/struct-iou.git
cd struct-iou
pip install . --upgrade
```

## Example Usage
```python
import structiou
# the ground-truth text parse tree, while the word boundaries are obtained by forced alignment
gold_tree = structiou.data.NLTKTree.fromstring(
    '( NT ( NT I ) ( NT ( NT am ) ( NT ( NT a ) ( NT cat ) ) ) )')
gold_word_boundaries = [
    ('I', 1.0, 1.5),
    ('am', 1.8, 2.0),
    ('a', 2.0, 2.2),
    ('cat', 2.2, 3.0)
]
gold_tree = structiou.data.ExtendedTree.from_tree_and_leaf_timestamps(
    gold_tree, gold_word_boundaries)
# the predicted speech parse tree, where the word identities are not necessary
predicted_tree = structiou.data.NLTKTree.fromstring(
    '( NT ( NT ( NT 1 ) ( NT 2 ) ) ( NT ( NT 3 ) ( NT ( NT 4 ) ( NT 5 ) ) ) )')
predicted_word_boundaries = [
    ('1', 1.0, 1.2),
    ('2', 1.2, 1.4),
    ('3', 1.8, 2.1),
    ('4', 2.1, 2.3),
    ('5', 2.3, 2.8)
]
predicted_tree = structiou.data.ExtendedTree.from_tree_and_leaf_timestamps(
    predicted_tree, predicted_word_boundaries)
# calculate the Struct-IoU, expected output: 0.6073
print(structiou.struct_iou(gold_tree, predicted_tree, flexible_terminal_alignment=True))
```


## Citation
If you use this code, please consider citing:
```
@misc{shi-etal-2024-structured,
    title = {Structured Tree Alignment for Evaluation of (Speech) Constituency Parsing},
    authors = {Shi, Freda, and Gimpel, Kevin and Livescu, Karen},
    year = {2024},
    eprint = {https://home.ttic.edu/~freda/data/papers/structaiou.pdf},
}
```
