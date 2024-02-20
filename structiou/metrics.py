import collections
import numpy as np
from typing import Set, Tuple

import structiou.utils as utils
import structiou.data as data


class TreeAligner(object):
    '''Tree aligner that aligns two trees by maximizing the sum of span iou of the aligned nodes.'''

    def __init__(self, ptree: data.ExtendedTree, gtree: data.ExtendedTree, threshold_iou: float = 0, flexible_terminal_alignment=True) -> None:
        '''Initialize the TreeAligner class.

        Args:
            ptree: The predicted tree.
            gtree: The gold tree.
            threshold_iou: The threshold of span iou to use. Defaults to 0.
            flexible_terminal_alignment: Whether to allow terminal nodes to be aligned with anything else. Defaults to True.
        '''
        self._ptree = ptree
        self._gtree = gtree
        self._threshold_iou = threshold_iou
        self._flexible_terminal_alignment = flexible_terminal_alignment
        self._n_ptree = ptree.n_nodes()
        self._n_gtree = gtree.n_nodes()
        span_ious = np.zeros((self._n_ptree, self._n_gtree))
        for i in range(self._n_ptree):
            for j in range(self._n_gtree):
                prange = self._ptree.index2node(i).time_range()
                grange = self._gtree.index2node(j).time_range()
                span_ious[i, j] = utils.span_iou(prange, grange)
        self._span_ious = span_ious * (span_ious > self._threshold_iou)
        self.max_matching_weights = -np.ones((self._n_ptree, self._n_gtree))
        self.trace = collections.defaultdict(set)

    def align(self, pnode: data.ExtendedTree, gnode: data.ExtendedTree, eps: float = 1e-6) -> Tuple[float, Set[Tuple[int, int]]]:
        '''Get the maximum matching weights of between pnode's subtree and gnode's subtree, given pnode aligns with gnode.

        Args:
            pnode: The predicted node.
            gnode: The gold node.
            eps: The epsilon to use. Defaults to 1e-6.

        Returns:
            The maximum matching weights of between pnode's subtree and gnode's subtree, given pnode aligns with gnode.
            The alignment trace, represented by a set of aligned node indices.
        '''
        # directly return the value if computed before
        pnode_index = pnode.index()
        gnode_index = gnode.index()
        if self.max_matching_weights[pnode_index, gnode_index] >= 0:
            return self.max_matching_weights[pnode_index, gnode_index], self.trace[pnode_index, gnode_index]
        # edge case: when there is no overlap or label unmatched, the max matching weight is 0
        if utils.eps_eq(self._span_ious[pnode_index, gnode_index], 0, eps) or (  # not overlapped
                not pnode.is_preterminal() and not gnode.is_preterminal() and pnode.label() != gnode.label()) or (  # NT label unmatched
                not self._flexible_terminal_alignment and pnode.label() != gnode.label()):  # terminal label unmatched under inflexible mode
            self.max_matching_weights[pnode_index, gnode_index] = 0
            self.trace[pnode_index, gnode_index] = set()
            return self.max_matching_weights[pnode_index, gnode_index], self.trace[pnode_index, gnode_index]
        # edge cases: when one of them is a leaf (preterminal): the max matching weight is the span iou between the two nodes
        if pnode.is_preterminal() or gnode.is_preterminal():
            self.max_matching_weights[pnode_index, gnode_index] = self._span_ious[pnode_index, gnode_index]
            self.trace[pnode_index, gnode_index] = {(pnode_index, gnode_index)}
            return self.max_matching_weights[pnode_index, gnode_index], self.trace[pnode_index, gnode_index]
        # compute the max matching weight from their non-overlapped descendants
        n_pnode_leaves = len(pnode.leaves())
        n_gnode_leaves = len(gnode.leaves())
        min_pnode_leaf_id = pnode.leaf_range()[0]
        min_gnode_leaf_id = gnode.leaf_range()[0]
        max_span_matching_weight = -1e10 * np.ones((n_pnode_leaves + 1, n_gnode_leaves + 1))
        max_span_matching_weight[0, 0] = 0
        trace = collections.defaultdict(set)
        for pnode_subtree_index in pnode.subtree_indices():
            if pnode_subtree_index == pnode_index:
                continue
            for gnode_subtree_index in gnode.subtree_indices():
                if gnode_subtree_index == gnode_index:
                    continue
                pnode_lleaf, pnode_rleaf = (x - min_pnode_leaf_id + 1 for x in pnode.index2node(pnode_subtree_index).leaf_range())
                gnode_lleaf, gnode_rleaf = (x - min_gnode_leaf_id + 1 for x in gnode.index2node(gnode_subtree_index).leaf_range())
                subtree_matching_weight, subtree_trace = self.align(pnode.index2node(pnode_subtree_index), gnode.index2node(gnode_subtree_index))
                for pnode_source_endpoint in range(pnode_lleaf):
                    for gnode_source_endpoint in range(gnode_lleaf):
                        update_value = max_span_matching_weight[pnode_source_endpoint, gnode_source_endpoint] + subtree_matching_weight
                        if update_value > max_span_matching_weight[pnode_rleaf, gnode_rleaf]:
                            max_span_matching_weight[pnode_rleaf, gnode_rleaf] = update_value
                            trace[pnode_rleaf, gnode_rleaf] = trace[pnode_source_endpoint, gnode_source_endpoint].copy()
                            trace[pnode_rleaf, gnode_rleaf].update(subtree_trace)
        for pnode_rleaf in range(max_span_matching_weight.shape[0]):
            for gnode_rleaf in range(max_span_matching_weight.shape[1]):
                if max_span_matching_weight[pnode_rleaf, gnode_rleaf] > self.max_matching_weights[pnode_index, gnode_index]:
                    self.max_matching_weights[pnode_index, gnode_index] = max_span_matching_weight[pnode_rleaf, gnode_rleaf]
                    self.trace[pnode_index, gnode_index] = trace[pnode_rleaf, gnode_rleaf].copy()
        # add the parents' span iou by definition
        self.max_matching_weights[pnode_index, gnode_index] += self._span_ious[pnode_index, gnode_index]
        self.trace[pnode_index, gnode_index].add((pnode_index, gnode_index))
        return self.max_matching_weights[pnode_index, gnode_index], self.trace[pnode_index, gnode_index]

    def __call__(self) -> Tuple[float, Set[Tuple[int, int]]]:
        '''Align the two trees. Since there is no requirements that the root nodes should be aligned, we enumerate all possible top-level alignments.

        Returns:
            The average IoU over the two trees.
            The alignment trace, represented by a set of aligned node indices.
        '''
        # first call: root nodes alignment as the baseline
        max_weight, max_weight_trace = self.align(self._ptree, self._gtree)
        # enumerate all possible top-level alignments
        n_pnode_leaves = len(self._ptree.leaves())
        n_gnode_leaves = len(self._gtree.leaves())
        max_span_matching_weight = -1e10 * np.ones((n_pnode_leaves + 1, n_gnode_leaves + 1))
        max_span_matching_weight[0, 0] = 0
        trace = collections.defaultdict(set)
        for pnode_subtree_index in self._ptree.subtree_indices():
            for gnode_subtree_index in self._gtree.subtree_indices():
                if pnode_subtree_index == 0 and gnode_subtree_index == 0:  # skip root node alignment
                    continue
                pnode_lleaf, pnode_rleaf = self._ptree.index2node(pnode_subtree_index).leaf_range()
                gnode_lleaf, gnode_rleaf = self._gtree.index2node(gnode_subtree_index).leaf_range()
                subtree_matching_weight, subtree_trace = self.align(self._ptree.index2node(pnode_subtree_index), self._gtree.index2node(gnode_subtree_index))
                for pnode_source_endpoint in range(pnode_lleaf):
                    for gnode_source_endpoint in range(gnode_lleaf):
                        update_value = max_span_matching_weight[pnode_source_endpoint, gnode_source_endpoint] + subtree_matching_weight
                        if update_value > max_span_matching_weight[pnode_rleaf, gnode_rleaf]:
                            max_span_matching_weight[pnode_rleaf, gnode_rleaf] = update_value
                            trace[pnode_rleaf, gnode_rleaf] = trace[pnode_source_endpoint, gnode_source_endpoint].copy()
                            trace[pnode_rleaf, gnode_rleaf].update(subtree_trace)
        for pnode_rleaf in range(max_span_matching_weight.shape[0]):
            for gnode_rleaf in range(max_span_matching_weight.shape[1]):
                if max_span_matching_weight[pnode_rleaf, gnode_rleaf] > max_weight:
                    max_weight = max_span_matching_weight[pnode_rleaf, gnode_rleaf]
                    max_weight_trace = trace[pnode_rleaf, gnode_rleaf].copy()
        struct_iou = max_weight * 2 / (self._ptree.n_nodes() + self._gtree.n_nodes())
        return struct_iou, max_weight_trace


def struct_iou(ptree: data.ExtendedTree, gtree: data.ExtendedTree, threshold_iou: float = 0, flexible_terminal_alignment=True) -> float:
    '''Calculate the structured intersection over union ratio of two trees.

    Args:
        ptree: The predicted tree.
        gtree: The gold tree.
        threshold_iou: The threshold of span iou to use. Defaults to 0.
        flexible_terminal_alignment: Whether to allow terminal nodes to be aligned with anything else. Defaults to True.

    Returns:
        The structured intersection over union ratio of the two trees.
    '''
    return TreeAligner(ptree, gtree, threshold_iou, flexible_terminal_alignment)()[0]
