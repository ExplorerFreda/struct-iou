import nltk
from typing import List, Generator, Tuple

import structiou.utils as utils


class NLTKTree(nltk.Tree):
    '''NLTK Tree class augmented with span collection method.'''

    def __init__(self, *args, **kwargs) -> None:
        '''Initialize the NLTKTree class.

        Args:
            *args: The arguments to pass to the nltk.Tree class.
            **kwargs: The keyword arguments to pass to the nltk.Tree class.
        '''
        super().__init__(*args, **kwargs)

    def spans(self, left: int = 0, label: bool = False, include_terminal: bool = False) -> Generator[int, int, str]:
        '''Get the spans of a tree.

        Args:
            left: The leftmost leaf index of the current node. Can be larger than 0 as the current node may character a subtree.
            label: Whether to include the label of the current node in the returned spans. Defaults to False.
            include_terminal: Whether to include the terminal nodes in the returned spans. Defaults to False.

        Yields:
            The spans of the tree rooted at node.
        '''
        if isinstance(self[0], str):
            assert len(self) == 1, 'Leaf node with multiple children detected.'
            if include_terminal:
                yield (left, left + 1, 'NT') if not label else (left, left + 1, self.label())
            return
        yield (left, left + len(self.leaves())) if not label else (left, left + len(self.leaves()), self.label())
        child_left = left
        for child in self:
            for x in child.spans(child_left, label, include_terminal):
                yield x
            n_child_leaves = len(child.leaves())
            child_left += n_child_leaves
        assert child_left == left + len(self.leaves()), 'Mismatch found in the construction of NLTKTree.'

    @classmethod
    def from_baretreestring(cls, string: str, nonterminal_label: str = 'N', preterminal_label: str = 'N') -> 'NLTKTree':
        '''Creates a tree from a string representation by first converting the string to a NLTK-readable one.

        Args:
            string: The string representation of the tree.
            nonterminal_label: The label to use for nonterminal nodes. Defaults to 'N'.
            preterminal_label: The label to use for preterminal nodes. Defaults to 'N'.
        '''
        string = utils.normalize_parentheses_string(string)
        augmented_st = ' '.join(map(lambda x: x if x in ['(', ')'] else f'(-PH {x})', string.split()))
        augmented_st = augmented_st.replace('( ', f'( {nonterminal_label} ')
        augmented_st = augmented_st.replace('(-PH ', f'( {preterminal_label} ')
        return cls.fromstring(augmented_st)


class ExtendedTree(NLTKTree):
    '''NLTK tree with additional timestamp information, and index to node maps.'''

    def __init__(self, index: int, time_range: Tuple[float, float], leaf_range: Tuple[float, float], *args, **kwargs) -> None:
        '''Initialize the ExtendedTree class.

        Args:
            index: The index of the current node (root of the current tree).
            time_range: The time range of the current node.
            leaf_range: The leaf index range (closed interval) covered by the current node.
            *args: The arguments to pass to the nltk.Tree class.
            **kwargs: The keyword arguments to pass to the nltk.Tree class.

        Returns:
            An instance of the ExtendedTree class.
        '''
        super().__init__(*args, **kwargs)
        self._index = index
        self._time_range = time_range
        self._leaf_range = leaf_range
        self._index2node = {index: self}
        for child in self:
            if hasattr(child, '_index2node'):
                self._index2node.update(child._index2node)

    def index2node(self, index : int) -> 'ExtendedTree':
        '''Get the node with the given index.

        Args:
            index: The index of the node to get.

        Returns:
            The node with the given index.
        '''
        return self._index2node[index]

    def index(self) -> int:
        '''Get the index of the current node.

        Returns:
            The index of the current node.
        '''
        return self._index

    def subtree_indices(self) -> List[int]:
        '''Get the indices of the subtrees of the current node.

        Returns:
            The indices of the subtrees of the current node.
        '''
        return sorted(self._index2node.keys())

    def leaf_range(self) -> Tuple[int, int]:
        '''Get the leaf range of the current node.

        Returns:
            The leaf range of the current node.
        '''
        return self._leaf_range

    def time_range(self) -> Tuple[float, float]:
        '''Get the time range of the current node.

        Returns:
            The time range of the current node.
        '''
        return self._time_range

    def n_nodes(self) -> int:
        '''Get the number of nodes in the current tree.

        Returns:
            The number of nodes in the current tree.
        '''
        if isinstance(self[0], str):
            assert len(self) == 1, 'Leaf node with multiple children detected.'
            return 1
        return sum(child.n_nodes() for child in self) + 1

    def is_preterminal(self) -> bool:
        '''Check whether the current node is a preterminal node.

        Returns:
            Whether the current node is a preterminal node.
        '''
        return len(self) == 1 and isinstance(self[0], str)

    @classmethod
    def from_tree_and_leaf_timestamps(cls, tree: NLTKTree, leaf_timestamps: List[Tuple[str, float, float]]) -> 'ExtendedTree':
        '''Create an ExtendedTree from a given tree and leaf timestamps.

        Args:
            tree: The tree to create the ExtendedTree from.
            leaf_timestamps: The leaf timestamps of the tree.

        Returns:
            The ExtendedTree created from the given tree and leaf timestamps.
        '''
        global_index = 0
        def construct_tree(node: NLTKTree, left: int, right: int) -> 'ExtendedTree':
            '''Construct an ExtendedTree from a given NLTKTree node.

            Args:
                node: The NLTKTree node to construct the ExtendedTree from.
                left: The leftmost leaf index of the current node.
                right: The rightmost (+1) leaf index of the current node.

            Returns:
                The ExtendedTree created from the given NLTKTree node.
            '''
            nonlocal global_index
            this_index = global_index
            global_index += 1
            current_left = left
            children = list()
            for child in node:
                if isinstance(child, str):  # leaf node
                    children.append(child)
                    current_left += 1
                else:  # non-leaf node
                    children.append(construct_tree(child, current_left, current_left + len(child.leaves())))
                    current_left += len(child.leaves())
            return ExtendedTree(
                this_index,
                (leaf_timestamps[left][-2], leaf_timestamps[right - 1][-1]),
                (left + 1, right),
                node.label(),
                children
            )
        return construct_tree(tree, 0, len(leaf_timestamps))

    def pformat(self, margin: int = 70, indent: int = 0, nodesep: str = '', parens: str = '()', quotes: bool = False) -> str:
        '''Pretty-print this tree. Overrides the Tree.pformat method.

        Args:
            margin: The right margin at which to do line-wrapping.
            indent: The indentation level at which printing begins. This number is used to decide how far to indent subsequent lines.
            nodesep: A string that is used to separate the node from the children.
            parens: A tuple specifying the two parentheses to use to.
            quotes: Whether use direct string or repr() to print the child nodes.

        Returns:
            A pretty-printed string representation of this tree.
        '''
        # Try writing it on one line.
        s = self._pformat_flat(nodesep, parens, quotes)
        if len(s) + indent < margin:
            return s

        # If it doesn't fit on one line, then write it on multi-lines.
        s = f'{parens[0]}{self._label}-{self._index}-{self._time_range[0]:.3f}~{self._time_range[1]:.3f}{nodesep}'
        for child in self:
            if isinstance(child, ExtendedTree):
                s += (
                    '\n'
                    + ' ' * (indent + 2)
                    + child.pformat(margin, indent + 2, nodesep, parens, quotes)
                )
            elif isinstance(child, tuple):
                s += '\n' + ' ' * (indent + 2) + "/".join(child)
            elif isinstance(child, str) and not quotes:
                s += '\n' + ' ' * (indent + 2) + "%s" % child
            else:
                s += '\n' + ' ' * (indent + 2) + repr(child)
        return s + parens[1]

    def _pformat_flat(self, nodesep: str, parens: str, quotes: bool) -> str:
        '''Special case for pretty print by printing the tree into one line.

        Args:
            nodesep: A string that is used to separate the node from the children.
            parens: A tuple specifying the two parentheses to use to.
            quotes: Whether use direct string or repr() to print the child nodes.

        Returns:
            A one-line pretty-printed string representation of this tree.
        '''
        childstrs = list()
        for child in self:
            if isinstance(child, ExtendedTree):
                childstrs.append(child._pformat_flat(nodesep, parens, quotes))
            elif isinstance(child, tuple):
                childstrs.append('/'.join(child))
            elif isinstance(child, str) and not quotes:
                childstrs.append(f'{child}')
            else:
                childstrs.append(repr(child))
        return f'{parens[0]}{self._label}-{self._index}-{self._time_range[0]:.3f}~{self._time_range[1]:.3f}{nodesep} {" ".join(childstrs)}{parens[1]}'
