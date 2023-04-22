from intervaltree import Interval, IntervalTree


class IntervalTreeHelper:
    def __init__(self,intervaltree):
        self.interval_tree = intervaltree

    def up_bnd_inclusive_at(self, p):
        """
        Rewrite the `IntervalTree.at()` function so that it's upperbound inclusive
        Returns the set of all intervals that contain p, will include the interval end.

        Completes in O(m + log n) time, where:
          * n = size of the tree
          * m = number of matches
        :rtype: set of Interval
        """
        root = self.interval_tree.top_node
        if not root:
            return set()
        return self.up_bnd_inclusive_search_point(root, p, set())

    def up_bnd_inclusive_search_point(self, root, point, result):
        """
        Rewrite the `IntervalTree.search_point()` function so that it's upperbound inclusive
        Returns all intervals that contain point. Interval's upper bound inclusive.
        """
        for k in root.s_center:
            if k.begin <= point <= k.end:
                result.add(k)
        if point < root.x_center and root[0]:
            return root[0].search_point(point, result)
        elif point > root.x_center and root[1]:
            return root[1].search_point(point, result)
        return result
