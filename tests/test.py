import structiou.data as data
import structiou.metrics as metrics
import structiou.utils as utils


EPS = 1e-4


def check_example(index:int, example: dict) -> None:
    '''Checks if the implemented algorithms work correctly on the given example.

    Args:
        example: The example to check.

    Raises:
        AssertionError: If the algorithm does not work correctly on the given example.
    '''
    predicted_tree = data.ExtendedTree.from_tree_and_leaf_timestamps(example['predicted_tree'], example['word_bounds'])
    gold_tree = data.ExtendedTree.from_tree_and_leaf_timestamps(example['gold_tree'], example['oracle_bounds'])
    struct_iou = metrics.TreeAligner(predicted_tree, gold_tree, flexible_terminal_alignment=example['flexible'])()[0]
    assert utils.eps_eq(struct_iou, example['gold_struct_iou'], EPS), f'Expected StructA-IoU {example["gold_struct_iou"]}, got {struct_iou} for Example {index}.'


if __name__ == '__main__':
    # example 1: normal alignment example: no distinction between preterminals and nonterminals
    example_1 = {
        'gold_tree': data.NLTKTree.fromstring('( NT ( NT I ) ( NT ( NT am ) ( NT ( NT a ) ( NT cat ) ) ) )'),
        'predicted_tree': data.NLTKTree.fromstring('( NT ( NT ( NT 1 ) ( NT 2 ) ) ( NT ( NT 3 ) ( NT ( NT 4 ) ( NT 5 ) ) ) )'),
        'oracle_bounds': [
            ('I', 1.0, 1.5),
            ('am', 1.8, 2.0),
            ('a', 2.0, 2.2),
            ('cat', 2.2, 3.0)
        ],
        'word_bounds': [
            ('1', 1.0, 1.2),
            ('2', 1.2, 1.4),
            ('3', 1.8, 2.1),
            ('4', 2.1, 2.3),
            ('5', 2.3, 2.8)
        ],
        'flexible': True,
        'gold_struct_iou': 0.607292,
    }

    # example 2: right-branching alignment example
    example_2 = {
        'gold_tree': data.NLTKTree.fromstring('( NT ( NT I ) ( NT ( NT am ) ( NT ( NT a ) ( NT cat ) ) ) )'),
        'predicted_tree': data.NLTKTree.fromstring('( NT ( NT 1 ) (NT ( NT 2 ) ( NT ( NT 3 ) ( NT ( NT 4 ) ( NT 5 ) ) ) ) )'),
        'oracle_bounds': [
            ('I', 1.0, 1.5),
            ('am', 1.8, 2.0),
            ('a', 2.0, 2.2),
            ('cat', 2.2, 3.0)
        ],
        'word_bounds': [
            ('1', 1.0, 1.2),
            ('2', 1.2, 1.4),
            ('3', 1.8, 2.1),
            ('4', 2.1, 2.3),
            ('5', 2.3, 2.8)
        ],
        'flexible': True,
        'gold_struct_iou': 0.557292,
    }

    # example 3: example with detailed labels
    example_3 = {
        'gold_tree': data.NLTKTree.fromstring('( S ( PRP I ) ( VP ( VBP am ) ( NP ( DT a ) ( NN cat ) ) ) )'),
        'predicted_tree': data.NLTKTree.fromstring('( S ( PRP ( X 1 ) ( PRP 2 ) ) ( VP ( VBP 3 ) ( NP ( DT 4 ) ( NN 5 ) ) ) )'),
        'oracle_bounds': [
            ('I', 1.0, 1.5),
            ('am', 1.8, 2.0),
            ('a', 2.0, 2.2),
            ('cat', 2.2, 3.0)
        ],
        'word_bounds': [
            ('1', 1.0, 1.2),
            ('2', 1.2, 1.4),
            ('3', 1.8, 2.1),
            ('4', 2.1, 2.3),
            ('5', 2.3, 2.8)
        ],
        'flexible': True,
        'gold_struct_iou': 0.607292,
    }

    # example 4: example with detailed labels, inflexible terminal alignment: PRP and X are not aligned
    example_4 = {
        'gold_tree': data.NLTKTree.fromstring('( S ( PRP I ) ( VP ( VBP am ) ( NP ( DT a ) ( NN cat ) ) ) )'),
        'predicted_tree': data.NLTKTree.fromstring('( S ( X ( X 1 ) ( PRP 2 ) ) ( VP ( VBP 3 ) ( NP ( DT 4 ) ( NN 5 ) ) ) )'),
        'oracle_bounds': [
            ('I', 1.0, 1.5),
            ('am', 1.8, 2.0),
            ('a', 2.0, 2.2),
            ('cat', 2.2, 3.0)
        ],
        'word_bounds': [
            ('1', 1.0, 1.2),
            ('2', 1.2, 1.4),
            ('3', 1.8, 2.1),
            ('4', 2.1, 2.3),
            ('5', 2.3, 2.8)
        ],
        'flexible': False,
        'gold_struct_iou': 0.557292,
    }

    examples = [example_1, example_2, example_3, example_4]
    for i, example in enumerate(examples):
        check_example(i, example)
